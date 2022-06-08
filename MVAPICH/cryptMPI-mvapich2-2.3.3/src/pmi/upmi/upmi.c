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
#include "upmi.h"
#include <stdlib.h>
#include <pthread.h>
#include <mpimem.h>
#include <unistd.h>
#include <sys/types.h>

struct PMI_keyval_t;
int _size, _rank, _appnum;
int _singleton_mode = 0;
static int _in_ibarrier = 0;
static int _in_iallgather = 0;
#if defined(HAVE_PMI2_IALLGATHER) && defined(HAVE_PMI2_IALLGATHER_WAIT)
static void * _iallgather_data = NULL;
static size_t _iallgather_data_size = 0;
#endif
pthread_mutex_t upmi_lock;
#ifdef USE_PMIX_API
static pmix_proc_t myproc;
pmix_status_t rc;
pmix_info_t info[1];

#define ANL_MAPPING "PMI_process_mapping"
#define PMI_MAX_VAL_LEN      4096            /* Maximum size of a PMI value */

static int convert_err(pmix_status_t rc)
{
    switch (rc) {
    case PMIX_ERR_INVALID_SIZE:
        return PMI_ERR_INVALID_SIZE;

    case PMIX_ERR_INVALID_KEYVALP:
        return PMI_ERR_INVALID_KEYVALP;

    case PMIX_ERR_INVALID_NUM_PARSED:
        return PMI_ERR_INVALID_NUM_PARSED;

    case PMIX_ERR_INVALID_ARGS:
        return PMI_ERR_INVALID_ARGS;

    case PMIX_ERR_INVALID_NUM_ARGS:
        return PMI_ERR_INVALID_NUM_ARGS;

    case PMIX_ERR_INVALID_LENGTH:
        return PMI_ERR_INVALID_LENGTH;

    case PMIX_ERR_INVALID_VAL_LENGTH:
        return PMI_ERR_INVALID_VAL_LENGTH;

    case PMIX_ERR_INVALID_VAL:
        return PMI_ERR_INVALID_VAL;

    case PMIX_ERR_INVALID_KEY_LENGTH:
        return PMI_ERR_INVALID_KEY_LENGTH;

    case PMIX_ERR_INVALID_KEY:
        return PMI_ERR_INVALID_KEY;

    case PMIX_ERR_INVALID_ARG:
        return PMI_ERR_INVALID_ARG;

    case PMIX_ERR_NOMEM:
        return PMI_ERR_NOMEM;

    case PMIX_ERR_UNPACK_READ_PAST_END_OF_BUFFER:
    case PMIX_ERR_LOST_CONNECTION_TO_SERVER:
    case PMIX_ERR_LOST_PEER_CONNECTION:
    case PMIX_ERR_LOST_CONNECTION_TO_CLIENT:
    case PMIX_ERR_NOT_SUPPORTED:
    case PMIX_ERR_NOT_FOUND:
    case PMIX_ERR_SERVER_NOT_AVAIL:
    case PMIX_ERR_INVALID_NAMESPACE:
    case PMIX_ERR_DATA_VALUE_NOT_FOUND:
    case PMIX_ERR_OUT_OF_RESOURCE:
    case PMIX_ERR_RESOURCE_BUSY:
    case PMIX_ERR_BAD_PARAM:
    case PMIX_ERR_IN_ERRNO:
    case PMIX_ERR_UNREACH:
    case PMIX_ERR_TIMEOUT:
    case PMIX_ERR_NO_PERMISSIONS:
    case PMIX_ERR_PACK_MISMATCH:
    case PMIX_ERR_PACK_FAILURE:
    case PMIX_ERR_UNPACK_FAILURE:
    case PMIX_ERR_UNPACK_INADEQUATE_SPACE:
    case PMIX_ERR_TYPE_MISMATCH:
    case PMIX_ERR_PROC_ENTRY_NOT_FOUND:
    case PMIX_ERR_UNKNOWN_DATA_TYPE:
    case PMIX_ERR_WOULD_BLOCK:
    case PMIX_EXISTS:
    case PMIX_ERROR:
        return PMI_FAIL;

    case PMIX_ERR_INIT:
        return PMI_ERR_INIT;

    case PMIX_SUCCESS:
        return PMI_SUCCESS;
    default:
        return PMI_FAIL;
    }
}
#endif

void UPMI_lock_init(void) {
    pthread_mutex_init(&upmi_lock, NULL);
}

void UPMI_lock_destroy(void) {
    pthread_mutex_destroy(&upmi_lock);
}

void UPMI_lock(void) {
    pthread_mutex_lock(&upmi_lock);
}

void UPMI_unlock(void) {
    pthread_mutex_unlock(&upmi_lock);
}

int UPMI_INIT( int *spawned ) {
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    rc = PMIx_Init(&myproc, NULL, 0);
    if(rc != PMIX_SUCCESS)
            return convert_err(rc);
    pmix_value_t *val;
    pmix_proc_t proc;
    pmix_info_t info[1];
    bool  val_optional = 1;
    memcpy(&proc, &myproc, sizeof(myproc));
    proc.rank = PMIX_RANK_WILDCARD;
    PMIX_INFO_CONSTRUCT(&info[0]);
    PMIX_INFO_LOAD(&info[0], PMIX_OPTIONAL, &val_optional, PMIX_BOOL);

    rc = PMIx_Get(&proc, PMIX_SPAWNED, info, 1, &val);
    if(rc != PMIX_SUCCESS) {
            *spawned = 0;
    }
    else
    {
            *spawned=val->data.flag;
    }
    return PMIX_SUCCESS;
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Init( spawned, &_size, &_rank, &_appnum );
    #else
    UPMI_lock_init();
    pmi_ret_val = PMI_Init( spawned );
    #endif
    return pmi_ret_val;
}

int UPMI_INITIALIZED( int *initialized ) { 
    #ifdef USE_PMIX_API
    *initialized = (PMIx_Initialized() ? UPMI_SUCCESS : UPMI_FAIL);
    #elif USE_PMI2_API
    *initialized = PMI2_Initialized();
    return UPMI_SUCCESS;
    #else
    return PMI_Initialized( initialized );
    #endif
}

int UPMI_FINALIZE( void ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    rc = PMIx_Finalize(NULL, 0);
    return convert_err(rc);
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Finalize();
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Finalize();
    UPMI_unlock();
    UPMI_lock_destroy();
    #endif
    return pmi_ret_val;
}

int UPMI_GET_SIZE( int *size ) { 
    #ifdef USE_PMIX_API
    pmix_value_t *val;
    pmix_info_t info[1];
    bool  val_optional = 1;
    pmix_proc_t proc;
    proc=myproc;
    proc.rank = PMIX_RANK_WILDCARD;
    PMIX_INFO_CONSTRUCT(&info[0]);
    PMIX_INFO_LOAD(&info[0], PMIX_OPTIONAL, &val_optional, PMIX_BOOL);
    if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, PMIX_JOB_SIZE, info, 1, &val))) {
        printf("Client ns %s rank %d: PMIx_Get job size failed: %d", myproc.nspace, myproc.rank, rc);
        abort();
    }
    *size= val->data.uint32;
    return convert_err(rc);
    #elif USE_PMI2_API
    *size = _size;
    return UPMI_SUCCESS;
    #else
    return PMI_Get_size( size );
    #endif
}

int UPMI_GET_RANK( int *rank ) { 
    #ifdef USE_PMIX_API
    *rank = myproc.rank;
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    *rank = _rank;
    return UPMI_SUCCESS;
    #else
    return PMI_Get_rank( rank );
    #endif
}

int UPMI_GET_APPNUM( int *appnum ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_value_t *val;
    pmix_info_t info[1];
    bool  val_optional = 1;
    pmix_proc_t proc;
    proc=myproc;
    proc.rank = PMIX_RANK_WILDCARD;
    PMIX_INFO_CONSTRUCT(&info[0]);
    PMIX_INFO_LOAD(&info[0], PMIX_OPTIONAL, &val_optional, PMIX_BOOL);

    if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, PMIX_APPNUM, info, 1, &val))) {
            *appnum=-1;
    } else {
    *appnum= val->data.uint32;
    }
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    *appnum = _appnum;
    pmi_ret_val = UPMI_SUCCESS;
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Get_appnum( appnum );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_GET_UNIVERSE_SIZE( int *size ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_value_t *val;
    pmix_info_t info[1];
    bool  val_optional = 1;
    pmix_proc_t proc = myproc;
    proc.rank = PMIX_RANK_WILDCARD;
    PMIX_INFO_CONSTRUCT(&info[0]);
    PMIX_INFO_LOAD(&info[0], PMIX_OPTIONAL, &val_optional, PMIX_BOOL);
    rc = PMIx_Get(&proc, PMIX_UNIV_SIZE, NULL, 0, &val);
    if (PMIX_SUCCESS == rc) {
            *size = val->data.uint32;
    }
    return convert_err(rc);
    #elif USE_PMI2_API
    char name[] = "universeSize";
    int outlen, found;
    PMI2_Info_GetJobAttrIntArray( name, size, sizeof (int), &outlen, &found );
    if( found && outlen==1 ) {
        pmi_ret_val = UPMI_SUCCESS;
    } else {
        pmi_ret_val = UPMI_FAIL;
    }
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Get_universe_size( size );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_BARRIER( void ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_info_t buf;
    int ninfo = 0;
    pmix_info_t *info = NULL;
    bool val = 1;
    info = &buf;
    PMIX_INFO_CONSTRUCT(info);
    PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &val, PMIX_BOOL);
    ninfo = 1;
    rc = PMIx_Fence(NULL, 0, info, ninfo);
    return convert_err(rc);
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_KVS_Fence();
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Barrier();
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_IBARRIER( void ) { 
    int rc;

    UPMI_lock();
    if (!_in_ibarrier) {
        _in_ibarrier = 1;
#ifdef USE_PMIX_API
    pmix_info_t buf;
    pmix_info_t *info = NULL;
    info = &buf;
    int ninfo = 1;
    rc = PMIx_Fence(NULL, 0, info, ninfo);
    return convert_err(rc);
#elif USE_PMI2_API
#   if defined(HAVE_PMI2_KVS_IFENCE) && defined(HAVE_PMI2_KVS_WAIT)
        rc = PMI2_KVS_Ifence();
#   else
        rc = PMI2_KVS_Fence();
#   endif
#else
#   if defined(HAVE_PMI_IBARRIER) && defined(HAVE_PMI_WAIT)
        rc = PMI_Ibarrier();
#   else
        rc = PMI_Barrier();
#   endif
#endif
    } else {
        rc = UPMI_FAIL;
    }
    UPMI_unlock();

    return rc;
}

int UPMI_WAIT( void ) { 
    int rc = UPMI_SUCCESS;

    UPMI_lock();
    if (_in_ibarrier) {
        _in_ibarrier = 0;
#ifdef USE_PMIX_API
        /* add code here*/
#elif USE_PMI2_API
#   if defined(HAVE_PMI2_KVS_IFENCE) && defined(HAVE_PMI2_KVS_WAIT)
        rc = PMI2_KVS_Wait();
#   endif
#else
#   if defined(HAVE_PMI_IBARRIER) && defined(HAVE_PMI_WAIT)
        rc = PMI_Wait();
#   endif
#endif
    } else {
        rc = UPMI_SUCCESS;
    }
    UPMI_unlock();

    return rc;
}

int UPMI_IALLGATHER( const char value[] ) {
    int rc;

    UPMI_lock();
    if (!_in_iallgather) {
        _in_iallgather = 1;
#ifdef USE_PMI2_API
#   if defined(HAVE_PMI2_SHMEM_IALLGATHER) \
        && defined(HAVE_PMI2_SHMEM_IALLGATHER_WAIT)
        rc = PMI2_SHMEM_Iallgather(value);
#   elif defined(HAVE_PMI2_IALLGATHER) && defined(HAVE_PMI2_IALLGATHER_WAIT)
        rc = PMI2_Iallgather(value);
        if (UPMI_SUCCESS == rc) {
            if (NULL == _iallgather_data) {
                _iallgather_data_size = _size * PMI2_MAX_VALLEN * sizeof(char);
                _iallgather_data = MPIU_Malloc(_iallgather_data_size);
            }

            if (NULL == _iallgather_data) {
                rc = UPMI_FAIL;
            } else {
                memset(_iallgather_data, 0, _iallgather_data_size);
            }
        }
#   else
        rc = UPMI_FAIL;
#   endif
#endif
    } else {
        rc = UPMI_FAIL;
    }
    UPMI_unlock();

    return rc;
}

int UPMI_IALLGATHER_WAIT( void **buf ) {
    int rc = UPMI_SUCCESS;

    UPMI_lock();
    if (_in_iallgather) {
        _in_iallgather = 0;
#ifdef USE_PMI2_API
#   if defined(HAVE_PMI2_SHMEM_IALLGATHER) \
        && defined(HAVE_PMI2_SHMEM_IALLGATHER_WAIT)
        rc = PMI2_SHMEM_Iallgather_wait(buf);
#   elif defined(HAVE_PMI2_IALLGATHER) && defined(HAVE_PMI2_IALLGATHER_WAIT)
        *buf = _iallgather_data;
        if (NULL != *buf) {
            rc = PMI2_Iallgather_wait(*buf);
        } else {
            rc = UPMI_FAIL;
        }
#   endif
#else
        rc = UPMI_FAIL;
#endif
    } else {
        rc = UPMI_SUCCESS;
    }
    UPMI_unlock();

    return rc;
}

int UPMI_IALLGATHER_FREE( void ) {
    int rc = UPMI_SUCCESS;

    UPMI_lock();
    if (!_in_iallgather) {
#ifdef USE_PMI2_API
#   if defined(HAVE_PMI2_SHMEM_IALLGATHER) \
        && defined(HAVE_PMI2_SHMEM_IALLGATHER_WAIT)
        /* nothing to do */
#   elif defined(HAVE_PMI2_IALLGATHER) && defined(HAVE_PMI2_IALLGATHER_WAIT)
        if (NULL != _iallgather_data) {
            MPIU_Free(_iallgather_data);
        }
#   endif
#endif
    } else {
        rc = UPMI_FAIL;
    }
    UPMI_unlock();

    return rc;
}

int UPMI_ABORT( int exit_code, const char error_msg[] ) { 
    #ifdef USE_PMIX_API
    rc = PMIx_Abort(exit_code, error_msg, NULL, 0);
    return convert_err(rc);
    #elif USE_PMI2_API
    return PMI2_Abort( 1, error_msg );    //flag = 1, abort all processes
    #else
    return PMI_Abort( exit_code, error_msg );
    #endif
}

int UPMI_KVS_GET_KEY_LENGTH_MAX( int *length ) { 
    #ifdef USE_PMIX_API
    *length = PMIX_MAX_KEYLEN;
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    *length = PMI2_MAX_KEYLEN;
    return UPMI_SUCCESS;
    #else
    return PMI_KVS_Get_key_length_max( length );
    #endif
}

int UPMI_KVS_GET_NAME_LENGTH_MAX( int *length ) { 
    #ifdef USE_PMIX_API
    *length = PMIX_MAX_NSLEN;
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    *length = PMI2_MAX_KEYLEN; //TODO is this correct?
    return UPMI_SUCCESS;
    #else
    return PMI_KVS_Get_name_length_max( length );
    #endif
}

int UPMI_KVS_GET_VALUE_LENGTH_MAX( int *length ) { 
    #ifdef USE_PMIX_API
    *length = PMI_MAX_VAL_LEN;
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    *length = PMI2_MAX_VALLEN;
    return UPMI_SUCCESS;
    #else
    return PMI_KVS_Get_value_length_max( length );
    #endif
}

int UPMI_KVS_GET_MY_NAME( char kvsname[], int length ) {
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_strncpy(kvsname, myproc.nspace, length-1);
    return UPMI_SUCCESS;
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Job_GetId( kvsname, length );
    if (pmi_ret_val == PMI2_ERR_OTHER && _size == 1) {
        _singleton_mode = 1;
        sprintf(kvsname, "singleton_kvs_%llu", getpid());
        pmi_ret_val = UPMI_SUCCESS;
    }
    #else
    UPMI_lock();
    pmi_ret_val = PMI_KVS_Get_my_name( kvsname, length );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_KVS_PUT( const char kvsname[], const char key[], const char value[] ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_value_t val;
    val.type = PMIX_STRING;
    val.data.string = (char*)value;
    rc = PMIx_Put(PMIX_GLOBAL, key, &val);
    return convert_err(rc);
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_KVS_Put( key, value );
    #else
    UPMI_lock();
    pmi_ret_val = PMI_KVS_Put( kvsname, key, value );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_KVS_GET( const char kvsname[], const char key[], char value[], int length ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_value_t *val;
    pmix_proc_t proc;
    if (!strcmp(key, ANL_MAPPING)) {
            proc = myproc;
            proc.rank = PMIX_RANK_WILDCARD;
            if (PMIX_SUCCESS == PMIx_Get(&proc, PMIX_ANL_MAP, NULL, 0, &val) &&
                    (NULL != val) && (PMIX_STRING == val->type)) {
                    pmix_strncpy(value, val->data.string, length-1);
                    return PMI_SUCCESS;
            } else {
                    return UPMI_FAIL;
            }
    }
    char *tmpkey=(char *) malloc(PMIX_MAX_KEYLEN);
    strcpy(tmpkey, key);
    char* token = strtok(tmpkey, "-");
    token = strtok(NULL, "-");
    proc.rank=atoi(token);
    rc = PMIx_Get(&proc, key, NULL, 0, &val);
    if (PMIX_SUCCESS == rc && NULL != val) {
            if (PMIX_STRING != val->type) {
                    rc = PMIX_ERROR;
            } else if (NULL != val->data.string) {
                    pmix_strncpy(value, val->data.string, length-1);
            }
    }
    return convert_err(rc);
    #elif USE_PMI2_API
    int vallen;
    pmi_ret_val = PMI2_KVS_Get( kvsname, PMI2_ID_NULL, key, value, length, &vallen );
    #else
    UPMI_lock();
    pmi_ret_val = PMI_KVS_Get( kvsname, key, value, length );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_KVS_COMMIT( const char kvsname[] ) { 
    #ifdef USE_PMIX_API
    rc = PMIx_Commit();
    return convert_err(rc);
    #elif USE_PMI2_API
    return UPMI_SUCCESS;
    #else
    return PMI_KVS_Commit( kvsname );
    #endif
}

int UPMI_PUBLISH_NAME( const char service_name[], const char port[], const struct MPID_Info *info_ptr ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_info_t info;
    pmix_strncpy(info.key, service_name, PMIX_MAX_KEYLEN);
    info.value.type = PMIX_STRING;
    info.value.data.string = (char*) port;
    rc = PMIx_Publish(&info, 1);
    return convert_err(rc);
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Nameserv_publish( service_name, info_ptr, port );
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Publish_name( service_name, port );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_UNPUBLISH_NAME( const char service_name[], const struct MPID_Info *info_ptr ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    char *keys[2];
    keys[0] = (char*) service_name;
    keys[1] = NULL;
    rc = PMIx_Unpublish(keys, NULL, 0);
    return convert_err(rc);
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Nameserv_unpublish( service_name, info_ptr );
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Unpublish_name( service_name );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_LOOKUP_NAME( const char service_name[], char port[], const struct MPID_Info *info_ptr ) { 
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    pmix_pdata_t pdata;
    PMIX_PDATA_CONSTRUCT(&pdata);
    pmix_strncpy(pdata.key, service_name, PMIX_MAX_KEYLEN);
    if (PMIX_SUCCESS != (rc = PMIx_Lookup(&pdata, 1, NULL, 0))) {
            return convert_err(rc);
            }
    if (PMIX_STRING != pdata.value.type || NULL == pdata.value.data.string) {
            return convert_err(PMIX_ERR_NOT_FOUND);
    }
    pmix_strncpy(port, pdata.value.data.string, PMIX_MAX_KEYLEN);
    return PMIX_SUCCESS;
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Nameserv_lookup( service_name, info_ptr, port, sizeof port );  
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Lookup_name( service_name, port );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

int UPMI_GET_NODE_ATTR( const char name[], char value[], int valuelen, int *found, int waitfor ) {
    #ifdef USE_PMI2_API
    return PMI2_Info_GetNodeAttr( name, value, valuelen, found, waitfor );
    #else
    return UPMI_FAIL;
    #endif
}

int UPMI_GET_NODE_ATTR_INT_ARRAY( const char name[], int array[], int arraylen, int *outlen, int *found ) {
    #ifdef USE_PMI2_API
    return PMI2_Info_GetNodeAttrIntArray( name, array, arraylen, outlen, found );
    #else
    return UPMI_FAIL;
    #endif
}

int UPMI_PUT_NODE_ATTR( const char name[], const char value[] ) {
    #ifdef USE_PMI2_API
    return PMI2_Info_PutNodeAttr( name, value );
    #else
    return UPMI_FAIL;
    #endif
}

int UPMI_GET_JOB_ATTR( const char name[], char value[], int valuelen, int *found ) {
    #ifdef USE_PMI2_API
    return PMI2_Info_GetJobAttr( name, value, valuelen, found );
    #else
    return UPMI_FAIL;
    #endif
}

int UPMI_GET_JOB_ATTR_INT_ARRAY( const char name[], int array[], int arraylen, int *outlen, int *found ) {
    #ifdef USE_PMI2_API
    return PMI2_Info_GetJobAttrIntArray( name, array, arraylen, outlen, found );
    #else
    return UPMI_FAIL;
    #endif
}

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
                   int errors[])
{
    int pmi_ret_val;
    #ifdef USE_PMIX_API
    /*add spawn code for pmix here*/
    #elif USE_PMI2_API
    pmi_ret_val = PMI2_Job_Spawn( count, cmds, argcs, argvs, maxprocs,
                           info_keyval_sizes, (const struct MPID_Info**)info_keyval_vectors,
                           preput_keyval_size, (const struct MPID_Info**)preput_keyval_vector,
                           jobId, jobIdSize, errors );
    #else
    UPMI_lock();
    pmi_ret_val = PMI_Spawn_multiple( count, cmds, argvs, maxprocs,
                               info_keyval_sizes, (const struct PMI_keyval_t**)info_keyval_vectors,
                               preput_keyval_size, (const struct PMI_keyval_t*)preput_keyval_vector,
                               errors );
    UPMI_unlock();
    #endif
    return pmi_ret_val;
}

