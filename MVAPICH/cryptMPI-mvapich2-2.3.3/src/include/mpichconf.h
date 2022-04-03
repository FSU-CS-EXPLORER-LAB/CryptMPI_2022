/* src/include/mpichconf.h.  Generated from mpichconf.h.in by configure.  */
/* src/include/mpichconf.h.in.  Generated from configure.ac by autoheader.  */

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* Copyright (c) 2001-2018, The Ohio State University. All rights
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
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#ifndef MPICHCONF_H_INCLUDED
#define MPICHCONF_H_INCLUDED


/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* The pamid assert level */
/* #undef ASSERT_LEVEL */

/* Define the number of CH3_RANK_BITS */
#define CH3_RANK_BITS 16

/* Define if using the mrail channel */
#define CHANNEL_MRAIL 1

/* Define if using the gen2 subchannel */
#define CHANNEL_MRAIL_GEN2 1

/* Define if using the nemesis channel */
/* #undef CHANNEL_NEMESIS */

/* Define if using the nemesis ib netmod */
/* #undef CHANNEL_NEMESIS_IB */

/* Define if using the psm channel */
/* #undef CHANNEL_PSM */

/* Define to enable Checkpoint/Restart support. */
/* #undef CKPT */

/* define to enable collection of statistics */
/* #undef COLLECT_STATS */

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* Define when using checkpoint aggregation */
/* #undef CR_AGGRE */

/* Define to enable FTB-CR support. */
/* #undef CR_FTB */

/* Define to 1 if using `alloca.c'. */
/* #undef C_ALLOCA */

/* Define the search path for machines files */
/* #undef DEFAULT_MACHINES_PATH */

/* Define the default remote shell program to use */
/* #undef DEFAULT_REMOTE_SHELL */

/* Define to disable use of ptmalloc. On Linux, disabling ptmalloc also
   disables registration caching. */
/* #undef DISABLE_PTMALLOC */

/* Define to disable shared-memory communication for debugging */
/* #undef ENABLED_NO_LOCAL */

/* Define to enable debugging mode where shared-memory communication is done
   only between even procs or odd procs */
/* #undef ENABLED_ODD_EVEN_CLIQUES */

/* Define to enable shared-memory collectives */
/* #undef ENABLED_SHM_COLLECTIVES */

/* Define to enable 3D Torus support */
/* #undef ENABLE_3DTORUS_SUPPORT */

/* Application checkpointing enabled */
/* #undef ENABLE_CHECKPOINTING */

/* define to add per-vc function pointers to override send and recv functions
   */
/* #undef ENABLE_COMM_OVERRIDES */

/* Define if FTB is enabled */
/* #undef ENABLE_FTB */

/* Enable site specific options for LLNL by default */
/* #undef ENABLE_LLNL_SITE_SPECIFIC_OPTIONS */

/* Define to 1 to enable memory-related MPI_T performance variables */
#define ENABLE_PVAR_MEM 0

/* Define to 1 to enable mvapich2-related MPI_T performance variables */
#define ENABLE_PVAR_MV2 0

/* Define to 1 to enable nemesis-related MPI_T performance variables */
#define ENABLE_PVAR_NEM 0

/* Define to 1 to enable message receive queue-related MPI_T performance
   variables */
#define ENABLE_PVAR_RECVQ 0

/* Define to 1 to enable rma-related MPI_T performance variables */
#define ENABLE_PVAR_RMA 0

/* Define to use SCR for checkpointing */
/* #undef ENABLE_SCR */

/* The value of false in Fortran */
#define F77_FALSE_VALUE 0

/* Fortran names are lowercase with no trailing underscore */
/* #undef F77_NAME_LOWER */

/* Fortran names are lowercase with two trailing underscores */
/* #undef F77_NAME_LOWER_2USCORE */

/* Fortran names are lowercase with two trailing underscores in stdcall */
/* #undef F77_NAME_LOWER_2USCORE_STDCALL */

/* Fortran names are lowercase with no trailing underscore in stdcall */
/* #undef F77_NAME_LOWER_STDCALL */

/* Fortran names are lowercase with one trailing underscore */
#define F77_NAME_LOWER_USCORE 1

/* Fortran names are lowercase with one trailing underscore in stdcall */
/* #undef F77_NAME_LOWER_USCORE_STDCALL */

/* Fortran names preserve the original case */
/* #undef F77_NAME_MIXED */

/* Fortran names preserve the original case in stdcall */
/* #undef F77_NAME_MIXED_STDCALL */

/* Fortran names preserve the original case with one trailing underscore */
/* #undef F77_NAME_MIXED_USCORE */

/* Fortran names preserve the original case with one trailing underscore in
   stdcall */
/* #undef F77_NAME_MIXED_USCORE_STDCALL */

/* Fortran names are uppercase */
/* #undef F77_NAME_UPPER */

/* Fortran names are uppercase in stdcall */
/* #undef F77_NAME_UPPER_STDCALL */

/* The value of true in Fortran */
#define F77_TRUE_VALUE 1

/* Define if we know the value of Fortran true and false */
#define F77_TRUE_VALUE_SET 1

/* Define FALSE */
#define FALSE 0

/* Directory to use in namepub */
/* #undef FILE_NAMEPUB_BASEDIR */

/* Define if using flux pmi client */
/* #undef FLUX_PMI_CLIENT */

/* Define if addresses are a different size than Fortran integers */
#define HAVE_AINT_DIFFERENT_THAN_FINT 1

/* Define if addresses are larger than Fortran integers */
#define HAVE_AINT_LARGER_THAN_FINT 1

/* Define to 1 if you have the `alarm' function. */
#define HAVE_ALARM 1

/* Define to 1 if you have `alloca', as a function or macro. */
#define HAVE_ALLOCA 1

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
#define HAVE_ALLOCA_H 1

/* Define if int32_t works with any alignment */
#define HAVE_ANY_INT32_T_ALIGNMENT 1

/* Define if int64_t works with any alignment */
#define HAVE_ANY_INT64_T_ALIGNMENT 1

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `bindprocessor' function. */
/* #undef HAVE_BINDPROCESSOR */

/* Define to 1 if the compiler supports __builtin_expect. */
#define HAVE_BUILTIN_EXPECT 1

/* define if the compiler defines __FUNC__ */
/* #undef HAVE_CAP__FUNC__ */

/* Define to 1 if you have the `CFUUIDCreate' function. */
/* #undef HAVE_CFUUIDCREATE */

/* Define to 1 if you have the `clock_getres' function. */
#define HAVE_CLOCK_GETRES 1

/* Define to 1 if you have the `clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if you have the <complex.h> header file. */
#define HAVE_COMPLEX_H 1

/* Define if CPU_SET and CPU_ZERO defined */
/* #undef HAVE_CPU_SET_MACROS */

/* Define if cpu_set_t is defined in sched.h */
#define HAVE_CPU_SET_T 1

/* Define to 1 if you have the <ctype.h> header file. */
#define HAVE_CTYPE_H 1

/* Define to 1 if you have the `cudaIpcGetMemHandle' function. */
/* #undef HAVE_CUDAIPCGETMEMHANDLE */

/* Define to enable CUDA IPC features */
/* #undef HAVE_CUDA_IPC */

/* Define if C++ is supported */
#define HAVE_CXX_BINDING 1

/* Define is C++ supports complex types */
#define HAVE_CXX_COMPLEX 1

/* define if the compiler supports exceptions */
#define HAVE_CXX_EXCEPTIONS /**/

/* Define if multiple __attribute__((alias)) are supported */
#define HAVE_C_MULTI_ATTR_ALIAS 1

/* Define if debugger support is included */
/* #undef HAVE_DEBUGGER_SUPPORT */

/* Define to 1 if you have the declaration of `RAI_FAMILY', and to 0 if you
   don't. */
/* #undef HAVE_DECL_RAI_FAMILY */

/* Define to 1 if you have the declaration of `strerror_r', and to 0 if you
   don't. */
#define HAVE_DECL_STRERROR_R 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Controls how alignment of doubles is performed, separate from other FP
   values */
/* #undef HAVE_DOUBLE_ALIGNMENT_EXCEPTION */

/* Controls how alignment is applied based on position of doubles in the
   structure */
/* #undef HAVE_DOUBLE_POS_ALIGNMENT */

/* Define to 1 if the system has the type `double _Complex'. */
#define HAVE_DOUBLE__COMPLEX 1

/* Define to 1 if you have the <endian.h> header file. */
#define HAVE_ENDIAN_H 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to enable error checking */
#define HAVE_ERROR_CHECKING MPID_ERROR_LEVEL_ALL

/* Define if environ extern is available */
/* #undef HAVE_EXTERN_ENVIRON */

/* Define to 1 to enable Fortran 2008 binding */
#define HAVE_F08_BINDING 0

/* Define to 1 if you have the <fcntl.h> header file. */
/* #undef HAVE_FCNTL_H */

/* Define if Fortran 90 type routines available */
#define HAVE_FC_TYPE_ROUTINES 1

/* Define to 1 if you have the `fdopen' function. */
#define HAVE_FDOPEN 1

/* Define if Fortran integer are the same size as C ints */
#define HAVE_FINT_IS_INT 1

/* Define to 1 if the system has the type `float _Complex'. */
#define HAVE_FLOAT__COMPLEX 1

/* Define if Fortran is supported */
#define HAVE_FORTRAN_BINDING 1

/* Define if GNU __attribute__ is supported */
#define HAVE_GCC_ATTRIBUTE 1

/* Define to 1 if you have the `gethostbyname' function. */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the `gethostname' function. */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have the `gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if you have the `getpid' function. */
#define HAVE_GETPID 1

/* Define to 1 if you have the `getsid' function. */
/* #undef HAVE_GETSID */

/* Define to 1 if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the `get_current_dir_name' function. */
#define HAVE_GET_CURRENT_DIR_NAME 1

/* Define if struct hostent contains h_addr_list */
#define HAVE_H_ADDR_LIST 1

/* Define to 1 if you have the `ibv_open_xrc_domain' function. */
/* #undef HAVE_IBV_OPEN_XRC_DOMAIN */

/* Define to 1 if you have the `inet_pton' function. */
#define HAVE_INET_PTON 1

/* Define if int16_t is supported by the C compiler */
#define HAVE_INT16_T 1

/* Define if int32_t is supported by the C compiler */
#define HAVE_INT32_T 1

/* Define if int64_t is supported by the C compiler */
#define HAVE_INT64_T 1

/* Define if int8_t is supported by the C compiler */
#define HAVE_INT8_T 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if struct iovec defined in sys/uio.h */
/* #undef HAVE_IOVEC_DEFINITION */

/* Define to 1 if you have the `isatty' function. */
/* #undef HAVE_ISATTY */

/* Define if you have the <knem_io.h> header file. */
/* #undef HAVE_KNEM_IO_H */

/* Define to 1 if you have the `cr' library (-lcr). */
/* #undef HAVE_LIBCR */

/* Define to 1 if you have the `fabric' library (-lfabric). */
/* #undef HAVE_LIBFABRIC */

/* Define to 1 if you have the `ftb' library (-lftb). */
/* #undef HAVE_LIBFTB */

/* Define to 1 if you have the `hcoll' library (-lhcoll). */
/* #undef HAVE_LIBHCOLL */

/* UMAD installation found. */
#define HAVE_LIBIBUMAD 1

/* Define to 1 if you have the `ibverbs' library (-libverbs). */
#define HAVE_LIBIBVERBS 1

/* Define to 1 if you have the `llc' library (-lllc). */
/* #undef HAVE_LIBLLC */

/* Define to 1 if you have the `memcached' library (-lmemcached). */
/* #undef HAVE_LIBMEMCACHED */

/* Define to 1 if you have the `mxm' library (-lmxm). */
/* #undef HAVE_LIBMXM */

/* Define to 1 if you have the `portals' library (-lportals). */
/* #undef HAVE_LIBPORTALS */

/* Define to 1 if you have the `psm2' library (-lpsm2). */
/* #undef HAVE_LIBPSM2 */

/* Define to 1 if you have the `psm_infinipath' library (-lpsm_infinipath). */
/* #undef HAVE_LIBPSM_INFINIPATH */

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Controls how alignment is applied based on position of long long ints in
   the structure */
/* #undef HAVE_LLINT_POS_ALIGNMENT */

/* Define if long double is supported */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if the system has the type `long double _Complex'. */
#define HAVE_LONG_DOUBLE__COMPLEX 1

/* Define if long long allowed */
#define HAVE_LONG_LONG 1

/* Define if long long is supported */
#define HAVE_LONG_LONG_INT 1

/* Define to 1 if you have the `mach_absolute_time' function. */
/* #undef HAVE_MACH_ABSOLUTE_TIME */

/* Define if C99-style variable argument list macro functionality */
#define HAVE_MACRO_VA_ARGS 1

/* Controls byte alignment of structs with doubles */
#define HAVE_MAX_DOUBLE_FP_ALIGNMENT 8

/* Controls byte alignment of structures with floats, doubles, and long
   doubles (for MPI structs) */
#define HAVE_MAX_FP_ALIGNMENT 16

/* Controls byte alignment of integer structures (for MPI structs) */
#define HAVE_MAX_INTEGER_ALIGNMENT 8

/* Controls byte alignment of structs with long doubles */
#define HAVE_MAX_LONG_DOUBLE_FP_ALIGNMENT 16

/* Controls byte alignment of structures (for aligning allocated structures)
   */
#define HAVE_MAX_STRUCT_ALIGNMENT 8

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `mkstemp' function. */
#define HAVE_MKSTEMP 1

/* Define to 1 if you have the `mmap' function. */
#define HAVE_MMAP 1

/* Define so that we can test whether the mpichconf.h file has been included
   */
#define HAVE_MPICHCONF 1

/* Define if the Fortran init code for MPI works from C programs without
   special libraries */
#define HAVE_MPI_F_INIT_WORKS_WITH_C 1

/* Define if multiple weak symbols may be defined */
#define HAVE_MULTIPLE_PRAGMA_WEAK 1

/* Define to 1 if you have the `munmap' function. */
#define HAVE_MUNMAP 1

/* Define if a name publishing service is available */
#define HAVE_NAMEPUB_SERVICE 1

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES /**/

/* define if the compiler implements namespace std */
#define HAVE_NAMESPACE_STD /**/

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define if the Fortran types are not available in C */
/* #undef HAVE_NO_FORTRAN_MPI_TYPES_IN_C */

/* Define to 1 if you have the <openacc.h> header file. */
/* #undef HAVE_OPENACC_H */

/* Define is the OSX thread affinity policy macros defined */
/* #undef HAVE_OSX_THREAD_AFFINITY */

/* Define if PAMI_CLIENT_MEMORY_OPTIMIZE is defined in pami.h */
/* #undef HAVE_PAMI_CLIENT_MEMORY_OPTIMIZE */

/* Define if PAMI_CLIENT_NONCONTIG is defined in pami.h */
/* #undef HAVE_PAMI_CLIENT_NONCONTIG */

/* Define if PAMI_GEOMETRY_MEMORY_OPTIMIZE is defined in pami.h */
/* #undef HAVE_PAMI_GEOMETRY_MEMORY_OPTIMIZE */

/* Define if PAMI_GEOMETRY_NONCONTIG is defined in pami.h */
/* #undef HAVE_PAMI_GEOMETRY_NONCONTIG */

/* Define if PAMI_IN_PLACE is defined in pami.h */
/* #undef HAVE_PAMI_IN_PLACE */

/* Define to 1 if you have the `PMI2_Iallgather' function. */
/* #undef HAVE_PMI2_IALLGATHER */

/* Define to 1 if you have the `PMI2_Iallgather_wait' function. */
/* #undef HAVE_PMI2_IALLGATHER_WAIT */

/* Define if pmi client supports PMI2_KVS_Ifence */
/* #undef HAVE_PMI2_KVS_IFENCE */

/* Define if pmi client supports PMI2_KVS_Wait */
/* #undef HAVE_PMI2_KVS_WAIT */

/* Define if pmi client supports PMI2_SHMEM_Iallgather */
/* #undef HAVE_PMI2_SHMEM_IALLGATHER */

/* Define if pmi client supports PMI2_SHMEM_Iallgather_wait */
/* #undef HAVE_PMI2_SHMEM_IALLGATHER_WAIT */

/* Define if pmi client supports PMI_Ibarrier */
/* #undef HAVE_PMI_IBARRIER */

/* Define if pmi client supports PMI_Wait */
/* #undef HAVE_PMI_WAIT */

/* Define to 1 if you have the <poll.h> header file. */
/* #undef HAVE_POLL_H */

/* Cray style weak pragma */
/* #undef HAVE_PRAGMA_CRI_DUP */

/* HP style weak pragma */
/* #undef HAVE_PRAGMA_HP_SEC_DEF */

/* Supports weak pragma */
#define HAVE_PRAGMA_WEAK 1

/* Define to 1 if you have the `process_vm_readv' function. */
#define HAVE_PROCESS_VM_READV 1

/* Define to 1 if you have the `pthread_cleanup_push' function. */
/* #undef HAVE_PTHREAD_CLEANUP_PUSH */

/* Define if pthread_cleanup_push is available, even as a macro */
/* #undef HAVE_PTHREAD_CLEANUP_PUSH_MACRO */

/* Define to 1 if you have the <pthread.h> header file. */
#define HAVE_PTHREAD_H 1

/* Define to 1 if you have the `pthread_yield' function. */
#define HAVE_PTHREAD_YIELD 1

/* Define to 1 if you have the `ptrace' function. */
/* #undef HAVE_PTRACE */

/* Define if ptrace parameters available */
/* #undef HAVE_PTRACE_CONT */

/* Define to 1 if you have the `putenv' function. */
#define HAVE_PUTENV 1

/* Define to 1 if you have the `qsort' function. */
#define HAVE_QSORT 1

/* Define to 1 if you have the `rand' function. */
/* #undef HAVE_RAND */

/* Define if ROMIO is enabled */
#define HAVE_ROMIO 1

/* Define to 1 if you have the `sched_getaffinity' function. */
#define HAVE_SCHED_GETAFFINITY 1

/* Define to 1 if you have the <sched.h> header file. */
#define HAVE_SCHED_H 1

/* Define to 1 if you have the `sched_setaffinity' function. */
#define HAVE_SCHED_SETAFFINITY 1

/* Define to 1 if you have the `sched_yield' function. */
#define HAVE_SCHED_YIELD 1

/* Define to 1 if you have the <search> header file. */
/* #undef HAVE_SEARCH */

/* Define to 1 if you have the `select' function. */
#define HAVE_SELECT 1

/* Define to 1 if you have the `setitimer' function. */
#define HAVE_SETITIMER 1

/* Define to 1 if you have the `setsid' function. */
/* #undef HAVE_SETSID */

/* Define to 1 if you have the `setsockopt' function. */
#define HAVE_SETSOCKOPT 1

/* Define to 1 if you have the `shmat' function. */
/* #undef HAVE_SHMAT */

/* Define to 1 if you have the `shmctl' function. */
/* #undef HAVE_SHMCTL */

/* Define to 1 if you have the `shmdt' function. */
/* #undef HAVE_SHMDT */

/* Define to 1 if you have the `shmget' function. */
/* #undef HAVE_SHMGET */

/* Define to 1 if you have the `sigaction' function. */
/* #undef HAVE_SIGACTION */

/* Define to 1 if you have the `signal' function. */
/* #undef HAVE_SIGNAL */

/* Define to 1 if you have the <signal.h> header file. */
/* #undef HAVE_SIGNAL_H */

/* Define to 1 if you have the `sigset' function. */
/* #undef HAVE_SIGSET */

/* Define to 1 if you have the `sleep' function. */
#define HAVE_SLEEP 1

/* Define to 1 if you have the `snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define to 1 if you have the `socket' function. */
#define HAVE_SOCKET 1

/* Define if socklen_t is available */
/* #undef HAVE_SOCKLEN_T */

/* Define to 1 if you have the `srand' function. */
/* #undef HAVE_SRAND */

/* Define to 1 if you have the <stdarg.h> header file. */
#define HAVE_STDARG_H 1

/* Define to 1 if you have the <stdbool.h> header file. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strdup' function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the `strerror' function. */
#define HAVE_STRERROR 1

/* Define to 1 if you have the `strerror_r' function. */
#define HAVE_STRERROR_R 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strncasecmp' function. */
#define HAVE_STRNCASECMP 1

/* Define to 1 if you have the `strndup' function. */
#define HAVE_STRNDUP 1

/* Define to 1 if you have the `strsignal' function. */
/* #undef HAVE_STRSIGNAL */

/* Define to 1 if you have the `strtoll' function. */
/* #undef HAVE_STRTOLL */

/* Define if struct ifconf can be used */
/* #undef HAVE_STRUCT_IFCONF */

/* Define if struct ifreq can be used */
/* #undef HAVE_STRUCT_IFREQ */

/* Define to 1 if you have the `syscall' function. */
#define HAVE_SYSCALL 1

/* Define to 1 if you have the <syscall.h> header file. */
/* #undef HAVE_SYSCALL_H */

/* Define if sys/bitypes.h exists */
#define HAVE_SYS_BITYPES_H 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
/* #undef HAVE_SYS_IOCTL_H */

/* Define to 1 if you have the <sys/ipc.h> header file. */
/* #undef HAVE_SYS_IPC_H */

/* Define to 1 if you have the <sys/mman.h> header file. */
/* #undef HAVE_SYS_MMAN_H */

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/poll.h> header file. */
/* #undef HAVE_SYS_POLL_H */

/* Define to 1 if you have the <sys/ptrace.h> header file. */
/* #undef HAVE_SYS_PTRACE_H */

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/shm.h> header file. */
/* #undef HAVE_SYS_SHM_H */

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
/* #undef HAVE_SYS_SOCKIO_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/syscall.h> header file. */
#define HAVE_SYS_SYSCALL_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H 1

/* Define to 1 if you have the <thread.h> header file. */
/* #undef HAVE_THREAD_H */

/* Define to 1 if you have the `thread_policy_set' function. */
/* #undef HAVE_THREAD_POLICY_SET */

/* Define to 1 if you have the `thr_yield' function. */
/* #undef HAVE_THR_YIELD */

/* Define to 1 if you have the `time' function. */
#define HAVE_TIME 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* define to enable timing collection */
/* #undef HAVE_TIMING */

/* Define to 1 if you have the `tsearch' function. */
/* #undef HAVE_TSEARCH */

/* Define if uint16_t is supported by the C compiler */
#define HAVE_UINT16_T 1

/* Define if uint32_t is supported by the C compiler */
#define HAVE_UINT32_T 1

/* Define if uint64_t is supported by the C compiler */
#define HAVE_UINT64_T 1

/* Define if uint8_t is supported by the C compiler */
#define HAVE_UINT8_T 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `unsetenv' function. */
/* #undef HAVE_UNSETENV */

/* Define to 1 if you have the `usleep' function. */
#define HAVE_USLEEP 1

/* Define to 1 if you have the `uuid_generate' function. */
/* #undef HAVE_UUID_GENERATE */

/* Define to 1 if you have the <uuid/uuid.h> header file. */
#define HAVE_UUID_UUID_H 1

/* Define if we have va_copy */
#define HAVE_VA_COPY 1

/* Define to 1 if you have the `vsnprintf' function. */
#define HAVE_VSNPRINTF 1

/* Define to 1 if you have the `vsprintf' function. */
#define HAVE_VSPRINTF 1

/* Define to 1 if you have the <wait.h> header file. */
/* #undef HAVE_WAIT_H */

/* Attribute style weak pragma */
#define HAVE_WEAK_ATTRIBUTE 1

/* Define to 1 if you have the `yield' function. */
/* #undef HAVE_YIELD */

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* define if the compiler defines __FUNCTION__ */
#define HAVE__FUNCTION__ /**/

/* define if the compiler defines __func__ */
#define HAVE__FUNC__ /**/

/* Define if we have __va_copy */
/* #undef HAVE___VA_COPY */

/* Define if using jsm pmi client */
/* #undef JSM_PMI_CLIENT */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_CPUID_RDTSC32 */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_CPUID_RDTSC64 */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_RDTSC */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_RDTSCP */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to enable checking of handles still allocated at MPI_Finalize */
/* #undef MPICH_DEBUG_HANDLEALLOC */

/* Define to enable handle checking */
/* #undef MPICH_DEBUG_HANDLES */

/* Define if each function exit should confirm memory arena correctness */
/* #undef MPICH_DEBUG_MEMARENA */

/* Define to enable preinitialization of memory used by structures and unions
   */
/* #undef MPICH_DEBUG_MEMINIT */

/* Define to enable mutex debugging */
/* #undef MPICH_DEBUG_MUTEX */

/* define to enable error messages */
#define MPICH_ERROR_MSG_LEVEL MPICH_ERROR_MSG_ALL

/* Define as the name of the debugger support library */
/* #undef MPICH_INFODLL_LOC */

/* MPICH is configured to require thread safety */
#define MPICH_IS_THREADED 1

/* Define to an expression that will result in an error checking mutex type.
   */
#define MPICH_PTHREAD_MUTEX_ERRORCHECK_VALUE PTHREAD_MUTEX_ERRORCHECK

/* Method used to implement atomic updates and access */
#define MPICH_THREAD_GRANULARITY MPICH_THREAD_GRANULARITY_GLOBAL

/* Level of thread support selected at compile time */
#define MPICH_THREAD_LEVEL MPI_THREAD_MULTIPLE

/* set to the name of the thread package */
#define MPICH_THREAD_PACKAGE_NAME MPICH_THREAD_PACKAGE_POSIX

/* If the compiler supports a TLS storage class define it to that here */
#define MPICH_TLS_SPECIFIER __thread

/* Define to enable channel rendezvous (Required by MVAPICH2) */
#define MPIDI_CH3_CHANNEL_RNDV 1

/* Define to turn on the inlining optimizations in Nemesis code */
/* #undef MPID_NEM_INLINE */

/* Method for local large message transfers. */
/* #undef MPID_NEM_LOCAL_LMT_IMPL */

/* Define to enable lock-free communication queues */
/* #undef MPID_NEM_USE_LOCK_FREE_QUEUES */

/* Define to enable use of sequence numbers (Required by MVAPICH2) */
#define MPID_USE_SEQUENCE_NUMBERS 1

/* Define if a port may be used to communicate with the processes */
/* #undef MPIEXEC_ALLOW_PORT */

/* Size of an MPI_STATUS, in Fortran, in Fortran integers */
#define MPIF_STATUS_SIZE 5

/* limits.h _MAX constant for MPI_Aint */
#define MPIR_AINT_MAX LONG_MAX

/* limits.h _MAX constant for MPI_Count */
#define MPIR_COUNT_MAX LLONG_MAX

/* a C type used to compute C++ bool reductions */
#define MPIR_CXX_BOOL_CTYPE _Bool

/* Define as the MPI Datatype handle for MPI::BOOL */
#define MPIR_CXX_BOOL_VALUE 0x4c000133

/* Define as the MPI Datatype handle for MPI::COMPLEX */
#define MPIR_CXX_COMPLEX_VALUE 0x4c000834

/* Define as the MPI Datatype handle for MPI::DOUBLE_COMPLEX */
#define MPIR_CXX_DOUBLE_COMPLEX_VALUE 0x4c001035

/* Define as the MPI Datatype handle for MPI::LONG_DOUBLE_COMPLEX */
#define MPIR_CXX_LONG_DOUBLE_COMPLEX_VALUE 0x4c002036

/* The C type for FORTRAN DOUBLE PRECISION */
#define MPIR_FC_DOUBLE_CTYPE double

/* The C type for FORTRAN REAL */
#define MPIR_FC_REAL_CTYPE float

/* C type to use for MPI_INTEGER16 */
/* #undef MPIR_INTEGER16_CTYPE */

/* C type to use for MPI_INTEGER1 */
#define MPIR_INTEGER1_CTYPE char

/* C type to use for MPI_INTEGER2 */
#define MPIR_INTEGER2_CTYPE short

/* C type to use for MPI_INTEGER4 */
#define MPIR_INTEGER4_CTYPE int

/* C type to use for MPI_INTEGER8 */
#define MPIR_INTEGER8_CTYPE long

/* limits.h _MAX constant for MPI_Offset */
#define MPIR_OFFSET_MAX LLONG_MAX

/* C type to use for MPI_REAL16 */
#define MPIR_REAL16_CTYPE long double

/* C type to use for MPI_REAL4 */
#define MPIR_REAL4_CTYPE float

/* C type to use for MPI_REAL8 */
#define MPIR_REAL8_CTYPE double

/* MPIR_Ucount is an unsigned MPI_Count-sized integer */
#define MPIR_Ucount unsigned long long

/* Method used to allocate MPI object handles */
#define MPIU_HANDLE_ALLOCATION_METHOD MPIU_HANDLE_ALLOCATION_MUTEX

/* Define to enable timing mutexes */
/* #undef MPIU_MUTEX_WAIT_TIME */

/* MPIU_PINT_FMT_DEC_SPEC is the format specifier for printing Pint as a
   decimal */
#define MPIU_PINT_FMT_DEC_SPEC "%ld"

/* MPIU_Pint is a pointer-sized integer */
#define MPIU_Pint long

/* Set to a type that can express the size of the entire address space */
#define MPIU_SIZE_T unsigned long

/* Method used to implement refcount updates */
#define MPIU_THREAD_REFCOUNT MPIU_REFCOUNT_NONE

/* define to disable reference counting predefined objects like MPI_COMM_WORLD
   */
/* #undef MPIU_THREAD_SUPPRESS_PREDEFINED_REFCOUNTS */

/* MPIU_UPINT_FMT_DEC_SPEC is the format specifier for printing Upint as a
   decimal */
#define MPIU_UPINT_FMT_DEC_SPEC "%lu"

/* MPIU_Upint is an unsigned pointer-sized integer */
#define MPIU_Upint unsigned long

/* Define to enable GEN2 interface */
#define MRAIL_GEN2_INTERFACE 1

/* Define to disable header caching. */
/* #undef MV2_DISABLE_HEADER_CACHING */

/* Set to current version of mvapich2 package */
#define MVAPICH2_VERSION "2.3.3"

/* Define if /bin must be in path */
/* #undef NEEDS_BIN_IN_PATH */

/* Define if environ decl needed */
/* #undef NEEDS_ENVIRON_DECL */

/* Define if fdopen needs a declaration */
#define NEEDS_FDOPEN_DECL 1

/* Define if gethostname needs a declaration */
/* #undef NEEDS_GETHOSTNAME_DECL */

/* Define if getsid needs a declaration */
/* #undef NEEDS_GETSID_DECL */

/* Define if mkstemp needs a declaration */
/* #undef NEEDS_MKSTEMP_DECL */

/* define if pointers must be aligned on pointer boundaries */
/* #undef NEEDS_POINTER_ALIGNMENT_ADJUST */

/* Define if _POSIX_SOURCE needed to get sigaction */
/* #undef NEEDS_POSIX_FOR_SIGACTION */

/* Define if pthread_mutexattr_settype needs a declaration */
/* #undef NEEDS_PTHREAD_MUTEXATTR_SETTYPE_DECL */

/* Define if putenv needs a declaration */
/* #undef NEEDS_PUTENV_DECL */

/* Define if snprintf needs a declaration */
/* #undef NEEDS_SNPRINTF_DECL */

/* Define if strdup needs a declaration */
/* #undef NEEDS_STRDUP_DECL */

/* Define if strerror_r needs a declaration */
/* #undef NEEDS_STRERROR_R_DECL */

/* Define if strict alignment memory access is required */
/* #undef NEEDS_STRICT_ALIGNMENT */

/* Define if strsignal needs a declaration */
/* #undef NEEDS_STRSIGNAL_DECL */

/* Define if sys/time.h is required to get timer definitions */
/* #undef NEEDS_SYS_TIME_H */

/* Define if usleep needs a declaration */
/* #undef NEEDS_USLEEP_DECL */

/* Define if vsnprintf needs a declaration */
/* #undef NEEDS_VSNPRINTF_DECL */

/* Name of package */
#define PACKAGE "mvapich2"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "mvapich-discuss@cse.ohio-state.edu"

/* Define to the full name of this package. */
#define PACKAGE_NAME "MVAPICH2"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "MVAPICH2 2.3.3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "mvapich2"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://mvapich.cse.ohio-state.edu"

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.3.3"

/* Define if PAMI_IN_PLACE is not defined in pami.h */
/* #undef PAMI_IN_PLACE */

/* Define to turn on the prefetching optimization in Nemesis code */
/* #undef PREFETCH_CELL */

/* Define to enable support from RDMA CM. */
#define RDMA_CM 1

/* The size of `bool', as computed by sizeof. */
#define SIZEOF_BOOL 1

/* The size of `char', as computed by sizeof. */
#define SIZEOF_CHAR 1

/* The size of `Complex', as computed by sizeof. */
#define SIZEOF_COMPLEX 8

/* The size of `double', as computed by sizeof. */
#define SIZEOF_DOUBLE 8

/* The size of `DoubleComplex', as computed by sizeof. */
#define SIZEOF_DOUBLECOMPLEX 16

/* The size of `double_int', as computed by sizeof. */
#define SIZEOF_DOUBLE_INT 16

/* The size of `double _Complex', as computed by sizeof. */
#define SIZEOF_DOUBLE__COMPLEX 16

/* Define size of PAC_TYPE_NAME */
#define SIZEOF_F77_DOUBLE_PRECISION 8

/* Define size of PAC_TYPE_NAME */
#define SIZEOF_F77_INTEGER 4

/* Define size of PAC_TYPE_NAME */
#define SIZEOF_F77_REAL 4

/* The size of `float', as computed by sizeof. */
#define SIZEOF_FLOAT 4

/* The size of `float_int', as computed by sizeof. */
#define SIZEOF_FLOAT_INT 8

/* The size of `float _Complex', as computed by sizeof. */
#define SIZEOF_FLOAT__COMPLEX 8

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* define if sizeof(int) = sizeof(MPI_Aint) */
/* #undef SIZEOF_INT_IS_AINT */

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 8

/* The size of `LongDoubleComplex', as computed by sizeof. */
#define SIZEOF_LONGDOUBLECOMPLEX 32

/* The size of `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE 16

/* The size of `long_double_int', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE_INT 32

/* The size of `long double _Complex', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE__COMPLEX 32

/* The size of `long_int', as computed by sizeof. */
#define SIZEOF_LONG_INT 16

/* The size of `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG 8

/* The size of `MPIR_Bsend_data_t', as computed by sizeof. */
#define SIZEOF_MPIR_BSEND_DATA_T 96

/* The size of `OPA_ptr_t', as computed by sizeof. */
#define SIZEOF_OPA_PTR_T 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `short_int', as computed by sizeof. */
#define SIZEOF_SHORT_INT 8

/* The size of `two_int', as computed by sizeof. */
#define SIZEOF_TWO_INT 8

/* The size of `unsigned char', as computed by sizeof. */
#define SIZEOF_UNSIGNED_CHAR 1

/* The size of `unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG 8

/* The size of `unsigned long long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG_LONG 8

/* The size of `unsigned short', as computed by sizeof. */
#define SIZEOF_UNSIGNED_SHORT 2

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* The size of `wchar_t', as computed by sizeof. */
#define SIZEOF_WCHAR_T 4

/* The size of `_Bool', as computed by sizeof. */
#define SIZEOF__BOOL 1

/* Define if using slurm pmi client */
/* #undef SLURM_PMI_CLIENT */

/* Define to specify the build OS type. */
/* #undef SOLARIS */

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at runtime.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
/* #undef STACK_DIRECTION */

/* Define calling convention */
#define STDCALL 

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if strerror_r returns char *. */
/* #undef STRERROR_R_CHAR_P */

/* Define TRUE */
#define TRUE 1

/* Define if MPI_Aint should be used instead of void * for storing attribute
   values */
/* #undef USE_AINT_FOR_ATTRVAL */

/* Define if alloca should be used if available */
/* #undef USE_ALLOCA */

/* Define if performing coverage tests */
/* #undef USE_COVERAGE */

/* Define to enable logging macros */
/* #undef USE_DBG_LOGGING */

/* Define to use the fastboxes in Nemesis code */
/* #undef USE_FASTBOX */

/* Define if file should be used for name publisher */
/* #undef USE_FILE_FOR_NAMEPUB */

/* Define if the length of a CHARACTER*(*) string in Fortran should be passed
   as size_t instead of int */
/* #undef USE_FORT_STR_LEN_SIZET */

/* Define to enable cuda kernel functions */
/* #undef USE_GPU_KERNEL */

/* Define to use ='s and spaces in the string utilities. */
/* #undef USE_HUMAN_READABLE_TOKENS */

/* define to choose logging library */
#define USE_LOGGING MPID_LOGGING_NONE

/* Define to enable memory tracing */
/* #undef USE_MEMORY_TRACING */

/* Define if we have sysv shared memory */
#define USE_MMAP_SHM 1

/* Define if mpiexec should create a new process group session */
/* #undef USE_NEW_SESSION */

/* Define if _POSIX_C_SOURCE needs to be undefined for struct ifconf */
/* #undef USE_NOPOSIX_FOR_IFCONF */

/* Define to use nothing to yield processor */
#define USE_NOTHING_FOR_YIELD 1

/* Define if PMI2 API must be used */
/* #undef USE_PMI2_API */

/* Define if PMIx client supports PMIx */
/* #undef USE_PMIX_API */

/* Define if access to PMI information through a port rather than just an fd
   is allowed */
#define USE_PMI_PORT 1

/* Define to enable use of rsh for command execution by default. */
/* #undef USE_RSH */

/* Define to use sched_yield to yield processor */
/* #undef USE_SCHED_YIELD_FOR_YIELD */

/* Define to use select to yield processor */
/* #undef USE_SELECT_FOR_YIELD */

/* Define if sigaction should be used to set signals */
/* #undef USE_SIGACTION */

/* Define if signal should be used to set signals */
/* #undef USE_SIGNAL */

/* Define to use sleep to yield processor */
/* #undef USE_SLEEP_FOR_YIELD */

/* Define it the socket verify macros should be enabled */
/* #undef USE_SOCK_VERIFY */

/* Define if _SVID_SOURCE needs to be defined for struct ifconf */
/* #undef USE_SVIDSOURCE_FOR_IFCONF */

/* Define if we have sysv shared memory */
/* #undef USE_SYSV_SHM */

/* Define if tsearch requires char pointers */
/* #undef USE_TSEARCH_WITH_CHARP */

/* Define to use usleep to yield processor */
/* #undef USE_USLEEP_FOR_YIELD */

/* Define if weak symbols should be used */
#define USE_WEAK_SYMBOLS 1

/* Define to use yield to yield processor */
/* #undef USE_YIELD_FOR_YIELD */

/* Version number of package */
#define VERSION "2.3.3"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define if words are little endian */
#define WORDS_LITTLEENDIAN 1

/* Define if configure will not tell us, for universal binaries */
/* #undef WORDS_UNIVERSAL_ENDIAN */

/* Define to 1 if `lex' declares `yytext' as a `char *' by default, not a
   `char[]'. */
/* #undef YYTEXT_POINTER */

/* Define to specify the build CPU is an AMD quad core. */
/* #undef _AMD_QUAD_CORE_ */

/* Define to specify the build CPU type. */
#define _EM64T_ 1

/* Define to enable MVAPICH2-GPU design. */
/* #undef _ENABLE_CUDA_ */

/* Define to enable hybrid design. */
#define _ENABLE_UD_ 1

/* Define to enable XRC support */
/* #undef _ENABLE_XRC_ */

/* Define to set the number of file offset bits */
/* #undef _FILE_OFFSET_BITS */

/* Define to specify the build CPU type. */
/* #undef _IA32_ */

/* Define to specify the build CPU type. */
/* #undef _IA64_ */

/* Define to enable Hardware multicast support. */
#define _MCST_SUPPORT_ 1 

/* Define to enable inter subnet communication support. */
/* #undef _MULTI_SUBNET_SUPPORT_ */

/* Define to enable the use of MVAPICH2 implmentation of collectives */
#define _OSU_COLLECTIVES_ 1

/* Define to enable MVAPICH2 customizations */
#define _OSU_MVAPICH_ 1

/* Define to enable switch IB-2 sharp support. */
/* #undef _SHARP_SUPPORT_ */

/* Define to enable intra-node communication via CMA */
#define _SMP_CMA_ 1

/* Define when using LiMIC2 */
/* #undef _SMP_LIMIC_ */

/* Define for Solaris 2.5.1 so the uint32_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT32_T */

/* Define for Solaris 2.5.1 so the uint64_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT64_T */

/* Define for Solaris 2.5.1 so the uint8_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT8_T */

/* Define to specify the build CPU type. */
/* #undef _X86_64_ */

/* define if bool is a built-in type */
/* #undef bool */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to the type of a signed integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int16_t */

/* Define to the type of a signed integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int32_t */

/* Define to the type of a signed integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int64_t */

/* Define to the type of a signed integer type of width exactly 8 bits if such
   a type exists and the standard includes do not define it. */
/* #undef int8_t */

/* Define to `int' if <sys/types.h> does not define. */
/* #undef pid_t */

/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#define restrict __restrict
/* Work around a bug in Sun C++: it does not support _Restrict or
   __restrict__, even though the corresponding Sun C compiler ends up with
   "#define restrict _Restrict" or "#define restrict __restrict__" in the
   previous line.  Perhaps some future version of Sun C++ will work with
   restrict; if so, hopefully it defines __RESTRICT like Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
# define __restrict__
#endif

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */

/* Define if socklen_t is not defined */
/* #undef socklen_t */

/* Define to the type of an unsigned integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint16_t */

/* Define to the type of an unsigned integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint32_t */

/* Define to the type of an unsigned integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint64_t */

/* Define to the type of an unsigned integer type of width exactly 8 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint8_t */

/* Define to empty if the keyword `volatile' does not work. Warning: valid
   code using `volatile' can become incorrect without. Disable with care. */
/* #undef volatile */


/* Include nopackage.h to undef autoconf-defined macros that cause conflicts in
 * subpackages.  This should not be necessary, but some packages are too
 * tightly intertwined right now (such as ROMIO and the MPICH core) */
#include "nopackage.h"
#include <mvapich.h>

#endif /* !defined(MPICHCONF_H_INCLUDED) */

