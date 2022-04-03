/* Basic platform-independent macro definitions for mutexes,
   thread-specific data and parameters for malloc. */

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

#ifndef _GENERIC_MALLOC_MACHINE_H
#define _GENERIC_MALLOC_MACHINE_H

/* <CHANNEL_MRAIL> */
/* #include <atomic.h> */
#include "atomic.h"
/* </CHANNEL_MRAIL> */

#ifndef mutex_init /* No threads, provide dummy macros */

# define NO_THREADS

/* The mutex functions used to do absolutely nothing, i.e. lock,
   trylock and unlock would always just return 0.  However, even
   without any concurrently active threads, a mutex can be used
   legitimately as an `in use' flag.  To make the code that is
   protected by a mutex async-signal safe, these macros would have to
   be based on atomic test-and-set operations, for example. */
typedef int mutex_t;

# define mutex_init(m)              (*(m) = 0)
# define mutex_lock(m)              ((*(m) = 1), 0)
# define mutex_trylock(m)           (*(m) ? 1 : ((*(m) = 1), 0))
# define mutex_unlock(m)            (*(m) = 0)

typedef void *tsd_key_t;
# define tsd_key_create(key, destr) do {} while(0)
# define tsd_setspecific(key, data) ((key) = (data))
# define tsd_getspecific(key, vptr) (vptr = (key))

# define thread_atfork(prepare, parent, child) do {} while(0)

#endif /* !defined mutex_init */

#ifndef atomic_full_barrier
# define atomic_full_barrier() __asm ("" ::: "memory")
#endif

#ifndef atomic_read_barrier
# define atomic_read_barrier() atomic_full_barrier ()
#endif

#ifndef atomic_write_barrier
# define atomic_write_barrier() atomic_full_barrier ()
#endif

#ifndef DEFAULT_TOP_PAD
# define DEFAULT_TOP_PAD 131072
#endif

#endif /* !defined(_GENERIC_MALLOC_MACHINE_H) */
