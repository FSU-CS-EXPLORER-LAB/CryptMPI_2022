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

#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <stdio.h>
#include <errno.h>
#include <string.h>
#if defined(_OSU_MVAPICH_) && defined(HAVE_LIBIBVERBS)
#include <infiniband/verbs.h>
#endif

// Define a prefix to distinguish output from different processes
// - prefix: string
extern void set_output_prefix( char* prefix );

// Get the output prefix
extern const char *get_output_prefix();

// Common print function
#define _COMMON_PRINT_( FMT, args... ) \
do { fprintf( stderr, "[%s][%s] "FMT, get_output_prefix(), __func__, ##args ); } while(0)


// Print (always) an error message 
#define PRINT_ERROR( FMT, args... ) \
do { _COMMON_PRINT_( FMT, ##args ); } while(0)

// Print (always) an error message with the errno error message
#if defined(_GNU_SOURCE)
#define MV2_STRERROR_R( ERRCODE, buf, BUF_SIZE, str )             \
  str = strerror_r( ERRCODE, buf, BUF_SIZE );
#else
#define MV2_STRERROR_R( ERRCODE, buf, BUF_SIZE, str )             \
  strerror_r( ERRCODE, buf, BUF_SIZE );                           \
  str = (char *) buf;
#endif

#define MAX_ERR_MSG 200
#define PRINT_ERROR_ERRNO( FMT, ERRCODE, args... )                \
do {                                                              \
  char err_msg[MAX_ERR_MSG], *str = NULL;                         \
  MV2_STRERROR_R( ERRCODE, err_msg, MAX_ERR_MSG, str );           \
  _COMMON_PRINT_( FMT": %s (%d)\n", ##args, str, ERRCODE );       \
} while(0)

// Check condition and if true, print the message
#define PRINT_INFO( COND, FMT, args... )  \
do {                                      \
    if ( COND ) {                         \
        _COMMON_PRINT_( FMT, ##args );    \
    }                                     \
} while(0)

// Check condition and if failed:
// - print an error message with the errno error message
// - abort
#define CHECK_ERROR_ERRNO( COND, FMT, errno, args... )   \
do {                                       \
    if( !(COND) ) {   \
        PRINT_ERROR_ERRNO( "At %s:%d: "FMT, errno, __FILE__, __LINE__, ##args );     \
        abort();                           \
}} while(0)


#if !defined(NDEBUG)

// (Only when debug enabled) 
// Check assertion and if failed:
// - print an error message 
// - abort
#define ASSERT_MSG( COND, FMT, args... )   \
do {                                       \
    if( !(COND) ) {   \
        _COMMON_PRINT_( "At %s:%d: "FMT, __FILE__, __LINE__, ##args );     \
        abort();                           \
}} while(0)


// (Only when debug enabled) 
// Check condition and if true, print the debug message 
#define PRINT_DEBUG( COND, FMT, args... ) \
do {                                      \
    if ( COND ) {   \
        _COMMON_PRINT_( FMT, ##args );    \
    }                                     \
} while(0)

#else

#define ASSERT_MSG( COND, FMT, args... )
#define PRINT_DEBUG( COND, FMT, args... )

#endif


// Verbosity level for sharp operations in collectives
extern int DEBUG_Sharp_verbose;


// Verbosity level for fork/kill/waitpid operations in mpirun_rsh and mpispawn
extern int DEBUG_Fork_verbose;

// Verbosity level for Fault Tolerance operations
extern int DEBUG_FT_verbose;

// Verbosity level for Checkpoint/Restart operations
extern int DEBUG_CR_verbose;

// Verbosity level for Migration operations
extern int DEBUG_MIG_verbose;

// Verbosity level for UD flow control
extern int DEBUG_UD_verbose;

// Verbosity level for UD ZCOPY Rndv
extern int DEBUG_ZCY_verbose;

// Verbosity level for On-Demand Connection Management
extern int DEBUG_CM_verbose;

// Verbosity level for XRC
extern int DEBUG_XRC_verbose;

// Verbosity level for UD stats
extern int DEBUG_UDSTAT_verbose;

// Verbosity level for memory stats
extern int DEBUG_MEM_verbose;

// Verbosity level for GPU CUDA
extern int DEBUG_CUDA_verbose;

// Verbosity level for IB MULTICAST
extern int DEBUG_MCST_verbose;

// Verbosity level for SHMEM Collectives
extern int DEBUG_SHM_verbose;

// Verbosity level for Channel manager
extern int DEBUG_CHM_verbose;

// Verbosity level for RNDV transfers
extern int DEBUG_RNDV_verbose;

// Verbosity level for Init phase
extern int DEBUG_INIT_verbose;

// Verbosity level for RDMA_CM
extern int DEBUG_RDMACM_verbose;

// Verbosity level for One-sided
extern int DEBUG_1SC_verbose;

// Verbosity level for dreg cache
extern int DEBUG_DREG_verbose;

// Initialize the verbosity level of the above variables
extern int initialize_debug_variables();

extern void mv2_print_mem_usage();

#endif
