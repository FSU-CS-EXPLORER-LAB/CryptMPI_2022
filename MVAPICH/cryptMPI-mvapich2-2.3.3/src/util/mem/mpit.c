/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#include <mpidimpl.h>
#include <mpimem.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <search.h>
#include <stdint.h>

#ifdef USE_MEMORY_TRACING
#   define mpit_malloc(a, line, file)           \
        MPIU_trmalloc(a, line, file)
#   define mpit_calloc(a, b, line, file)        \
        MPIU_trcalloc(a, b, line, file)
#   define mpit_free(a, line, file)             \
        MPIU_trfree(a, line, file)
#   define mpit_strdup(a, line, file)           \
        MPIU_trstrdup(a, line, file)
#   define mpit_realloc(a, b, line, file)       \
        MPIU_trrealloc(a, b, line, file)
#   define mpit_memalign(a, b, c, line, file)   \
        posix_memalign(a, b, c)
#else
#   define mpit_malloc(a, line, file)           \
        malloc((size_t)a)
#   define mpit_calloc(a, b, line, file)        \
        calloc((size_t)a, (size_t)b)
#   define mpit_free(a, line, file)             \
        free((void *)a)
#   define mpit_strdup(a, line, file)           \
        strdup(a)
#   define mpit_realloc(a, b, line, file)       \
        realloc((void *)a, (size_t)b)
#   define mpit_memalign(a, b, c, line, file)   \
        posix_memalign(a, b, c)
#endif

MPIR_T_PVAR_ULONG2_LEVEL_DECL_STATIC(MV2, mem_allocated);
MPIR_T_PVAR_ULONG2_HIGHWATERMARK_DECL_STATIC(MV2, mem_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_malloc_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_calloc_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_memalign_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_strdup_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_realloc_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_free_calls);
MPIR_T_PVAR_ULONG_COUNTER_DECL_STATIC(MV2, num_memalign_free_calls)

typedef struct {
    void * addr;
    size_t size;
} MPIT_MEMORY_T;

static void * oracle = NULL;

/*
 * This variable is used to count memory before MPIT is initialized
 */
static size_t unaccounted = 0;
static size_t unaccounted_malloc = 0;
static size_t unaccounted_calloc = 0;
static size_t unaccounted_memalign = 0;
static size_t unaccounted_strdup = 0;
static size_t unaccounted_realloc = 0;
static size_t unaccounted_free = 0;
static size_t unaccounted_memalign_free = 0;

static int initialized = 0;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t oracle_mutex = PTHREAD_MUTEX_INITIALIZER;

extern int mv2_enable_pvar_mem;

void Real_Free (void *ptr);

static inline void
increment_malloc_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_malloc_calls, 1);
    }

    else {
        unaccounted_malloc++;
    }
 
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_calloc_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END
   
    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_calloc_calls, 1);
    }

    else {
        unaccounted_calloc++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_realloc_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_realloc_calls, 1);
    }

    else {
        unaccounted_realloc++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_free_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_free_calls, 1);
    }

    else {
        unaccounted_free++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_memalign_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_memalign_calls, 1);
    }

    else {
        unaccounted_memalign++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_memalign_free_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_memalign_free_calls, 1);
    }

    else {
        unaccounted_memalign_free++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

static inline void
increment_strdup_counter (void)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_COUNTER_INC(MV2, num_strdup_calls, 1);
    }

    else {
        unaccounted_strdup++;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}


static int
ptr_cmp (const void * mptr1, const void * mptr2)
{
    uintptr_t addr1 = (uintptr_t)((MPIT_MEMORY_T *)mptr1)->addr;
    uintptr_t addr2 = (uintptr_t)((MPIT_MEMORY_T *)mptr2)->addr;

    if (addr1 == addr2) {
        return 0;
    }

    return addr1 < addr2 ? -1 : 1;
}

#if 0
/* 
 * This function is used for debugging purposes only
 */
static void
ptr_print (const void * node, const VISIT which, const int depth)
{
    MPIT_MEMORY_T * data;
    int i = 0;

    switch (which) {
        case preorder:
            break;
        case postorder:
            data = *(MPIT_MEMORY_T **)node;
            for (i = 0; i < depth; i++) printf("*");
            printf("[%p: %ld]\n", data->addr, data->size);
            fflush(stdout);
            break;
        case endorder:
            break;
        case leaf:
            data = *(MPIT_MEMORY_T **)node;
            for (i = 0; i < depth; i++) printf("*");
            printf("[%p: %ld]\n", data->addr, data->size);
            fflush(stdout);
            break;
    }
}
#endif

static MPIT_MEMORY_T *
oracle_insert (void * ptr, size_t size)
{
    MPIT_MEMORY_T * mptr = mpit_malloc(sizeof (MPIT_MEMORY_T), __LINE__,
            __FILE__); 
    void * result;

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&oracle_mutex);
    MPIU_THREAD_CHECK_END

    if (mptr) {
        mptr->addr = ptr;
        mptr->size = size;
        result = tsearch(mptr, &oracle, ptr_cmp);
        mptr = result ? *(MPIT_MEMORY_T **)result : NULL;
        fflush(stdout);
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&oracle_mutex);
    MPIU_THREAD_CHECK_END

    return mptr;
}

static MPIT_MEMORY_T *
oracle_find (void * ptr)
{
    MPIT_MEMORY_T m = { .addr = ptr };
    MPIT_MEMORY_T * mptr;
    void * result;

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&oracle_mutex);
    MPIU_THREAD_CHECK_END
    result = tfind(&m, &oracle, ptr_cmp);
    mptr = result ? *(MPIT_MEMORY_T **)result : NULL;
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&oracle_mutex);
    MPIU_THREAD_CHECK_END

    return mptr;
}

static void
oracle_delete (MPIT_MEMORY_T * ptr)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&oracle_mutex);
    MPIU_THREAD_CHECK_END
    tdelete(ptr, &oracle, ptr_cmp);
    mpit_free(ptr, __LINE__, __FILE__);
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&oracle_mutex);
    MPIU_THREAD_CHECK_END
}

static inline void
increment_counter (signed long size)
{
    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END

    if (initialized) {
        MPIR_T_PVAR_LEVEL_INC(MV2, mem_allocated, size);
        MPIR_T_PVAR_ULONG2_HIGHWATERMARK_UPDATE(MV2, mem_allocated,
                PVAR_LEVEL_mem_allocated);
    }

    else {
        unaccounted += size;
    }

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

void
MPIT_MEM_REGISTER_PVARS (void)
{
    MPIR_T_PVAR_LEVEL_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG_LONG,
            mem_allocated,
            0, /* initial value */
            MPI_T_VERBOSITY_USER_BASIC,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "CH3", /* category name */
            "Current level of allocated memory within the MPI library");
    MPIR_T_PVAR_HIGHWATERMARK_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG_LONG,
            mem_allocated,
            0, /* initial value */
            MPI_T_VERBOSITY_USER_BASIC,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "CH3", /* category name */
            "Maximum level of memory ever allocated within the MPI library");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_malloc_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_malloc calls");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_calloc_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_calloc calls");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_memalign_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_memalign calls");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_strdup_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_strdup calls");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_realloc_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_realloc calls");

    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_free_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_free calls");
    MPIR_T_PVAR_COUNTER_REGISTER_STATIC(
            MV2,
            MPI_UNSIGNED_LONG,
            num_memalign_free_calls,
            MPI_T_VERBOSITY_TUNER_DETAIL,
            MPI_T_BIND_NO_OBJECT,
            (MPIR_T_PVAR_FLAG_READONLY | MPIR_T_PVAR_FLAG_CONTINUOUS),
            "MEM", /* category name */
            "Number of MPIT_memalign_free calls");

    MPIU_THREAD_CHECK_BEGIN 
    pthread_mutex_lock(&mutex);
    MPIU_THREAD_CHECK_END
    initialized = 1;
    MPIR_T_PVAR_LEVEL_INC(MV2, mem_allocated, unaccounted);
    MPIR_T_PVAR_ULONG2_HIGHWATERMARK_UPDATE(MV2, mem_allocated,
            PVAR_LEVEL_mem_allocated);

    MPIR_T_PVAR_COUNTER_INC(MV2, num_malloc_calls, unaccounted_malloc);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_calloc_calls, unaccounted_calloc);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_memalign_calls, unaccounted_memalign);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_strdup_calls, unaccounted_strdup);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_realloc_calls, unaccounted_realloc);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_free_calls, unaccounted_free);
    MPIR_T_PVAR_COUNTER_INC(MV2, num_memalign_free_calls,
            unaccounted_memalign_free);

    MPIU_THREAD_CHECK_BEGIN
    pthread_mutex_unlock(&mutex);
    MPIU_THREAD_CHECK_END}

void *
MPIT_malloc (size_t size, int lineno, char const * filename)
{
    void *ptr;

    if(mv2_enable_pvar_mem)	
    {
        ptr = mpit_malloc(size, lineno, filename);

        if (ptr) {
            increment_counter(size);
            oracle_insert(ptr, size);
        }    
    }
    else
    {
        ptr = mpit_malloc(size, lineno, filename);
    }

    return ptr;
}

void *
MPIT_calloc (size_t nelements, size_t elementSize, int lineno, char const *filename)
{
    void *ptr;

    if(mv2_enable_pvar_mem)
    {
        increment_calloc_counter();
        ptr = mpit_calloc(nelements, elementSize, lineno, filename);
 
        size_t size = nelements * elementSize;

        if (ptr) {
           increment_counter(size);
           oracle_insert(ptr, size);
        }
    }
    else
    {
        ptr = mpit_calloc(nelements, elementSize, lineno, filename);
    }      

    return ptr;
}

int
MPIT_memalign (void ** ptr, size_t alignment, size_t size, int lineno, char
        const * filename)
{
    int rv;

    if(mv2_enable_pvar_mem)
    {
        increment_memalign_counter();
    
        rv = mpit_memalign(ptr, alignment, size, lineno, filename);
   
        if (!rv) {
            increment_counter(size);
            oracle_insert(*ptr, size);
        }
    }
    else
    {
        rv = mpit_memalign(ptr, alignment, size, lineno, filename);
    }
    return rv;
}

char *
MPIT_strdup (const char * s, int lineno, char const * filename)
{
    char *ptr;

    if(mv2_enable_pvar_mem)
    {
        increment_strdup_counter();
        ptr = mpit_strdup(s, lineno, filename);
        size_t size = strlen(s);
        if (ptr) {
            increment_counter(size);
            oracle_insert(ptr, size);
        }
    }
    else
    {
        ptr = mpit_strdup(s, lineno, filename);        
    }

    return ptr;
}

void *
MPIT_realloc (void * ptr, size_t size, int lineno, char const * filename)
{
    if(mv2_enable_pvar_mem)
    {
        increment_realloc_counter();
    
        if (ptr) 
        {
            MPIT_MEMORY_T * mptr = oracle_find(ptr);
            size_t oldsize;
            MPIU_Assert(NULL != mptr);
            oldsize = mptr->size;
            ptr = mpit_realloc(ptr, size, lineno, filename);
            if (ptr) {
                oracle_delete(mptr);
                oracle_insert(ptr, size);
                increment_counter(size - oldsize);
            }
            else if (!size) {
                oracle_delete(mptr);
                increment_counter(size - oldsize);
            }
        }
        else 
        {
            ptr = mpit_realloc(ptr, size, lineno, filename);
            if (ptr) {
                oracle_insert(ptr, size);
                increment_counter(size);
            }
        }
    }
    else
    {
        ptr = mpit_realloc(ptr, size, lineno, filename);
    }
    return ptr;
}

void
MPIT_free (void * ptr, int lineno, char const * filename)
{
    if(mv2_enable_pvar_mem)
    {
        increment_free_counter();
        size_t oldsize = 0;

        if (ptr) {
            MPIT_MEMORY_T * mptr = oracle_find(ptr);

            if (mptr) {
                oldsize = mptr->size;
                oracle_delete(mptr);
            }
        }

        mpit_free(ptr, lineno, filename);
        increment_counter(0 - oldsize);        
    }
    else
    {
        mpit_free(ptr, lineno, filename);
    }    
}

void
MPIT_memalign_free (void * ptr, int lineno, char const * filename)
{
    if(mv2_enable_pvar_mem)
    {
        increment_memalign_free_counter();
        size_t oldsize = 0;

        if (ptr) {
            MPIT_MEMORY_T * mptr = oracle_find(ptr);

            if (mptr) {
                oldsize = mptr->size;
                oracle_delete(mptr);
            }
        }

        Real_Free(ptr);
        increment_counter(0 - oldsize); 
    }
    else
    {
        Real_Free(ptr);
    }
}

void
MPIT_shmdt (void * ptr, int lineno, char const * filename)
{
    if(mv2_enable_pvar_mem)
    {
        increment_memalign_free_counter();
        size_t oldsize = 0;

        if (ptr) {
            MPIT_MEMORY_T * mptr = oracle_find(ptr);

            if (mptr) {
                oldsize = mptr->size;
                oracle_delete(mptr);
            }
        }

        shmdt(ptr);
        increment_counter(0 - oldsize);  
    }
    else
    {
        shmdt(ptr);
    }

}
