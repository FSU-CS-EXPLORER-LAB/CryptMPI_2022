/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#ifndef UPMI_H_INCLUDED
#define UPMI_H_INCLUDED

#include "mpichconf.h"

#ifdef USE_PMIX_API
    #include <pmix.h>
    #include <pmi.h>
#elif USE_PMI2_API
    #include <pmi2.h>
#else
    #include <pmi.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define UPMI_SUCCESS                0
#define UPMI_FAIL                   -1
#define UPMI_ERR_INIT               1
#define UPMI_ERR_NOMEM              2
#define UPMI_ERR_INVALID_ARG        3
#define UPMI_ERR_INVALID_KEY        4
#define UPMI_ERR_INVALID_KEY_LENGTH 5
#define UPMI_ERR_INVALID_VAL        6
#define UPMI_ERR_INVALID_VAL_LENGTH 7
#define UPMI_ERR_INVALID_LENGTH     8
#define UPMI_ERR_INVALID_NUM_ARGS   9
#define UPMI_ERR_INVALID_ARGS       10
#define UPMI_ERR_INVALID_NUM_PARSED 11
#define UPMI_ERR_INVALID_KEYVALP    12
#define UPMI_ERR_INVALID_SIZE       13
#define UPMI_ERR_OTHER              14

struct MPID_Info;

int UPMI_INIT( int *spawned );

int UPMI_INITIALIZED( int *initialized );

int UPMI_FINALIZE( void );

int UPMI_GET_SIZE( int *size );

int UPMI_GET_RANK( int *rank );

int UPMI_GET_APPNUM( int *appnum );

int UPMI_GET_UNIVERSE_SIZE( int *size );

int UPMI_BARRIER( void );

int UPMI_IBARRIER( void );

int UPMI_WAIT( void );

int UPMI_IALLGATHER( const char value[] );

int UPMI_IALLGATHER_WAIT( void **buf );

int UPMI_IALLGATHER_FREE( void );

int UPMI_ABORT( int exit_code, const char error_msg[] );

int UPMI_KVS_GET_KEY_LENGTH_MAX( int *length );

int UPMI_KVS_GET_NAME_LENGTH_MAX( int *length );

int UPMI_KVS_GET_VALUE_LENGTH_MAX( int *length );

int UPMI_KVS_GET_MY_NAME( char kvsname[], int length );

int UPMI_KVS_PUT( const char kvsname[], const char key[], const char value[] );

int UPMI_KVS_GET( const char kvsname[], const char key[], char value[], int length );

int UPMI_KVS_COMMIT( const char kvsname[] );

int UPMI_PUBLISH_NAME( const char service_name[], const char port[], const struct MPID_Info *info_ptr );

int UPMI_UNPUBLISH_NAME( const char service_name[], const struct MPID_Info *info_ptr );

int UPMI_LOOKUP_NAME( const char service_name[], char port[], const struct MPID_Info *info_ptr );

int UPMI_GET_NODE_ATTR( const char name[], char value[], int valuelen, int *found, int waitfor );

int UPMI_GET_NODE_ATTR_INT_ARRAY( const char name[], int array[], int arraylen, int *outlen, int *found );

int UPMI_PUT_NODE_ATTR( const char name[], const char value[] );

int UPMI_GET_JOB_ATTR( const char name[], char value[], int valuelen, int *found );

int UPMI_GET_JOB_ATTR_INT_ARRAY( const char name[], int array[], int arraylen, int *outlen, int *found );

int UPMI_JOB_SPAWN(int count,
                   const char * cmds[],
                   int argcs[],
                   const char ** argvs[],
                   const int maxprocs[],
                   const int info_keyval_sizes[],
                   const void *info_keyval_vectors[],
                   int preput_keyval_size,
                   const void *preput_keyval_vector,
                   char jobId[],
                   int jobIdSize,
                   int errors[]);

#if defined(__cplusplus)
}
#endif

#endif //UPMI_H_INCLUDED
