/* src/include/mpichconf.h.  Generated from mpichconf.h.in by configure.  */
/* src/include/mpichconf.h.in.  Generated from configure.ac by autoheader.  */

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#ifndef MPICHCONF_H_INCLUDED
#define MPICHCONF_H_INCLUDED


/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define the number of CH3_RANK_BITS */
#define CH3_RANK_BITS 16

/* Define the number of CH4_RANK_BITS */
/* #undef CH4_RANK_BITS */

/* define to enable collection of statistics */
/* #undef COLLECT_STATS */

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* Define to 1 if using `alloca.c'. */
/* #undef C_ALLOCA */

/* Define the search path for machines files */
/* #undef DEFAULT_MACHINES_PATH */

/* Define the default remote shell program to use */
/* #undef DEFAULT_REMOTE_SHELL */

/* Define to disable shared-memory communication for debugging */
/* #undef ENABLED_NO_LOCAL */

/* Define to enable debugging mode where shared-memory communication is done
   only between even procs or odd procs */
/* #undef ENABLED_ODD_EVEN_CLIQUES */

/* Define to enable shared-memory collectives */
/* #undef ENABLED_SHM_COLLECTIVES */

/* Application checkpointing enabled */
/* #undef ENABLE_CHECKPOINTING */

/* define to add per-vc function pointers to override send and recv functions
   */
/* #undef ENABLE_COMM_OVERRIDES */

/* Define if FTB is enabled */
/* #undef ENABLE_FTB */

/* Define to enable using Izem CPU atomics */
/* #undef ENABLE_IZEM_ATOMIC */

/* Define to enable using Izem queues */
/* #undef ENABLE_IZEM_QUEUE */

/* Define to enable using Izem locks and condition variables */
/* #undef ENABLE_IZEM_SYNC */

/* Define to 1 to enable getdims-related MPI_T performance variables */
#define ENABLE_PVAR_DIMS 0

/* Define to 1 to enable nemesis-related MPI_T performance variables */
#define ENABLE_PVAR_NEM 0

/* Define to 1 to enable message receive queue-related MPI_T performance
   variables */
#define ENABLE_PVAR_RECVQ 0

/* Define to 1 to enable rma-related MPI_T performance variables */
#define ENABLE_PVAR_RMA 0

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

/* Define to 1 if the system has the type `CACHE_DESCRIPTOR'. */
/* #undef HAVE_CACHE_DESCRIPTOR */

/* Define to 1 if the system has the type `CACHE_RELATIONSHIP'. */
/* #undef HAVE_CACHE_RELATIONSHIP */

/* define if the compiler defines __FUNC__ */
/* #undef HAVE_CAP__FUNC__ */

/* Define to 1 if you have the `CFUUIDCreate' function. */
/* #undef HAVE_CFUUIDCREATE */

/* OFI netmod is built */
/* #undef HAVE_CH4_NETMOD_OFI */

/* Portals4 netmod is built */
/* #undef HAVE_CH4_NETMOD_PORTALS4 */

/* UCX netmod is built */
/* #undef HAVE_CH4_NETMOD_UCX */

/* Define to 1 if you have the `clz' function. */
/* #undef HAVE_CLZ */

/* Define to 1 if you have the `clzl' function. */
/* #undef HAVE_CLZL */

/* Define to 1 if you have the <CL/cl_ext.h> header file. */
/* #undef HAVE_CL_CL_EXT_H */

/* Define to 1 if you have the <complex.h> header file. */
#define HAVE_COMPLEX_H 1

/* Define to 1 if you have the `cpuset_setaffinity' function. */
/* #undef HAVE_CPUSET_SETAFFINITY */

/* Define to 1 if you have the `cpuset_setid' function. */
/* #undef HAVE_CPUSET_SETID */

/* Define if CPU_SET and CPU_ZERO defined */
#define HAVE_CPU_SET_MACROS 1

/* Define if cpu_set_t is defined in sched.h */
#define HAVE_CPU_SET_T 1

/* Define to 1 if you have the <ctype.h> header file. */
#define HAVE_CTYPE_H 1

/* Define to 1 if we have -lcuda */
/* #undef HAVE_CUDA */

/* Define to 1 if you have the <cuda.h> header file. */
/* #undef HAVE_CUDA_H */

/* Define to 1 if you have the <cuda_runtime_api.h> header file. */
/* #undef HAVE_CUDA_RUNTIME_API_H */

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

/* Define to 1 if you have the declaration of `CTL_HW', and to 0 if you don't.
   */
#define HAVE_DECL_CTL_HW 0

/* Define to 1 if you have the declaration of `fabsf', and to 0 if you don't.
   */
#define HAVE_DECL_FABSF 1

/* Define to 1 if you have the declaration of `getexecname', and to 0 if you
   don't. */
#define HAVE_DECL_GETEXECNAME 0

/* Define to 1 if you have the declaration of `GetModuleFileName', and to 0 if
   you don't. */
#define HAVE_DECL_GETMODULEFILENAME 0

/* Define to 1 if you have the declaration of `getprogname', and to 0 if you
   don't. */
#define HAVE_DECL_GETPROGNAME 0

/* Define to 1 if you have the declaration of `HW_NCPU', and to 0 if you
   don't. */
#define HAVE_DECL_HW_NCPU 0

/* Define to 1 if you have the declaration of `lgrp_latency_cookie', and to 0
   if you don't. */
/* #undef HAVE_DECL_LGRP_LATENCY_COOKIE */

/* Define to 1 if you have the declaration of
   `nvmlDeviceGetMaxPcieLinkGeneration', and to 0 if you don't. */
/* #undef HAVE_DECL_NVMLDEVICEGETMAXPCIELINKGENERATION */

/* Define to 1 if you have the declaration of `pthread_getaffinity_np', and to
   0 if you don't. */
#define HAVE_DECL_PTHREAD_GETAFFINITY_NP 1

/* Define to 1 if you have the declaration of `pthread_setaffinity_np', and to
   0 if you don't. */
#define HAVE_DECL_PTHREAD_SETAFFINITY_NP 1

/* Embedded mode; just assume we do not have Valgrind support */
#define HAVE_DECL_RUNNING_ON_VALGRIND 0

/* Define to 1 if you have the declaration of `sched_getcpu', and to 0 if you
   don't. */
#define HAVE_DECL_SCHED_GETCPU 1

/* Define to 1 if you have the declaration of `snprintf', and to 0 if you
   don't. */
#define HAVE_DECL_SNPRINTF 1

/* Define to 1 if you have the declaration of `strcasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRCASECMP 1

/* Define to 1 if you have the declaration of `strerror_r', and to 0 if you
   don't. */
#define HAVE_DECL_STRERROR_R 1

/* Define to 1 if you have the declaration of `strtoull', and to 0 if you
   don't. */
#define HAVE_DECL_STRTOULL 1

/* Define to 1 if you have the declaration of `_putenv', and to 0 if you
   don't. */
#define HAVE_DECL__PUTENV 0

/* Define to 1 if you have the declaration of `_SC_LARGE_PAGESIZE', and to 0
   if you don't. */
#define HAVE_DECL__SC_LARGE_PAGESIZE 0

/* Define to 1 if you have the declaration of `_SC_NPROCESSORS_CONF', and to 0
   if you don't. */
#define HAVE_DECL__SC_NPROCESSORS_CONF 1

/* Define to 1 if you have the declaration of `_SC_NPROCESSORS_ONLN', and to 0
   if you don't. */
#define HAVE_DECL__SC_NPROCESSORS_ONLN 1

/* Define to 1 if you have the declaration of `_SC_NPROC_CONF', and to 0 if
   you don't. */
#define HAVE_DECL__SC_NPROC_CONF 0

/* Define to 1 if you have the declaration of `_SC_NPROC_ONLN', and to 0 if
   you don't. */
#define HAVE_DECL__SC_NPROC_ONLN 0

/* Define to 1 if you have the declaration of `_SC_PAGESIZE', and to 0 if you
   don't. */
#define HAVE_DECL__SC_PAGESIZE 1

/* Define to 1 if you have the declaration of `_SC_PAGE_SIZE', and to 0 if you
   don't. */
#define HAVE_DECL__SC_PAGE_SIZE 1

/* Define to 1 if you have the declaration of `_strdup', and to 0 if you
   don't. */
#define HAVE_DECL__STRDUP 0

/* Define to 1 if you have the <dirent.h> header file. */
#define HAVE_DIRENT_H 1

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

/* Define to 1 if you have the `ffs' function. */
#define HAVE_FFS 1

/* Define to 1 if you have the `ffsl' function. */
#define HAVE_FFSL 1

/* Define if Fortran integer are the same size as C ints */
#define HAVE_FINT_IS_INT 1

/* Define to 1 if the system has the type `float _Complex'. */
#define HAVE_FLOAT__COMPLEX 1

/* Define to 1 if you have the `fls' function. */
/* #undef HAVE_FLS */

/* Define to 1 if you have the `flsl' function. */
/* #undef HAVE_FLSL */

/* Define if Fortran is supported */
#define HAVE_FORTRAN_BINDING 1

/* Define if GNU __attribute__ is supported */
#define HAVE_GCC_ATTRIBUTE 1

/* Define to 1 if you have the `gethostname' function. */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `getsid' function. */
/* #undef HAVE_GETSID */

/* Define to 1 if the system has the type `GROUP_AFFINITY'. */
/* #undef HAVE_GROUP_AFFINITY */

/* Define to 1 if the system has the type `GROUP_RELATIONSHIP'. */
/* #undef HAVE_GROUP_RELATIONSHIP */

/* Define to 1 if you have the `host_info' function. */
/* #undef HAVE_HOST_INFO */

/* Define if hwloc is available */
#define HAVE_HWLOC 1

/* Define to 1 if you have the <hwloc.h> header file. */
/* #undef HAVE_HWLOC_H */

/* Define if struct hostent contains h_addr_list */
#define HAVE_H_ADDR_LIST 1

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

/* Define to 1 if the system has the type `KAFFINITY'. */
/* #undef HAVE_KAFFINITY */

/* Define if you have the <knem_io.h> header file. */
/* #undef HAVE_KNEM_IO_H */

/* Define to 1 if you have the <kstat.h> header file. */
/* #undef HAVE_KSTAT_H */

/* Define to 1 if you have the `cr' library (-lcr). */
/* #undef HAVE_LIBCR */

/* Define to 1 if you have the `fabric' library (-lfabric). */
/* #undef HAVE_LIBFABRIC */

/* Define to 1 if you have the `ftb' library (-lftb). */
/* #undef HAVE_LIBFTB */

/* Define to 1 if we have -lgdi32 */
/* #undef HAVE_LIBGDI32 */

/* Define to 1 if you have the `hcoll' library (-lhcoll). */
/* #undef HAVE_LIBHCOLL */

/* Define to 1 if you have the `hwloc' library (-lhwloc). */
/* #undef HAVE_LIBHWLOC */

/* Define to 1 if you have the `ibverbs' library (-libverbs). */
/* #undef HAVE_LIBIBVERBS */

/* Define to 1 if we have -lkstat */
/* #undef HAVE_LIBKSTAT */

/* Define to 1 if we have -llgrp */
/* #undef HAVE_LIBLGRP */

/* Define to 1 if you have the `llc' library (-lllc). */
/* #undef HAVE_LIBLLC */

/* Define to 1 if you have the `memcached' library (-lmemcached). */
/* #undef HAVE_LIBMEMCACHED */

/* Define to 1 if you have the `mxm' library (-lmxm). */
/* #undef HAVE_LIBMXM */

/* Define to 1 if you have the `pmi' library (-lpmi). */
/* #undef HAVE_LIBPMI */

/* Define to 1 if you have the `pmix' library (-lpmix). */
/* #undef HAVE_LIBPMIX */

/* Define to 1 if you have the `portals' library (-lportals). */
/* #undef HAVE_LIBPORTALS */

/* Define to 1 if you have the `ucp' library (-lucp). */
/* #undef HAVE_LIBUCP */

/* Define to 1 if you have the <libudev.h> header file. */
/* #undef HAVE_LIBUDEV_H */

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Controls how alignment is applied based on position of long long ints in
   the structure */
/* #undef HAVE_LLINT_POS_ALIGNMENT */

/* Define to 1 if the system has the type `LOGICAL_PROCESSOR_RELATIONSHIP'. */
/* #undef HAVE_LOGICAL_PROCESSOR_RELATIONSHIP */

/* Define if long double is supported */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if the system has the type `long double _Complex'. */
#define HAVE_LONG_DOUBLE__COMPLEX 1

/* Define if long long allowed */
#define HAVE_LONG_LONG 1

/* Define if long long is supported */
#define HAVE_LONG_LONG_INT 1

/* Define to 1 if you have the <mach/mach_host.h> header file. */
/* #undef HAVE_MACH_MACH_HOST_H */

/* Define to 1 if you have the <mach/mach_init.h> header file. */
/* #undef HAVE_MACH_MACH_INIT_H */

/* Define if C99-style variable argument list macro functionality */
#define HAVE_MACRO_VA_ARGS 1

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

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

/* Define to 1 if you have the `memalign' function. */
#define HAVE_MEMALIGN 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `mkstemp' function. */
#define HAVE_MKSTEMP 1

/* Define so that we can test whether the mpichconf.h file has been included
   */
#define HAVE_MPICHCONF 1

/* Define if the Fortran init code for MPI works from C programs without
   special libraries */
#define HAVE_MPI_F_INIT_WORKS_WITH_C 1

/* Define if multiple weak symbols may be defined */
#define HAVE_MULTIPLE_PRAGMA_WEAK 1

/* Define if a name publishing service is available */
#define HAVE_NAMEPUB_SERVICE 1

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES /**/

/* define if the compiler implements namespace std */
#define HAVE_NAMESPACE_STD /**/

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define if netinet/in.h exists */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

/* Define if netloc is available in either user specified path or in system
   path */
/* #undef HAVE_NETLOC */

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define if the Fortran types are not available in C */
/* #undef HAVE_NO_FORTRAN_MPI_TYPES_IN_C */

/* Define to 1 if the system has the type `NUMA_NODE_RELATIONSHIP'. */
/* #undef HAVE_NUMA_NODE_RELATIONSHIP */

/* Define to 1 if you have the <NVCtrl/NVCtrl.h> header file. */
/* #undef HAVE_NVCTRL_NVCTRL_H */

/* Define to 1 if you have the <nvml.h> header file. */
/* #undef HAVE_NVML_H */

/* Define to 1 if you have the `openat' function. */
#define HAVE_OPENAT 1

/* Define to 1 if you have the <OpenCL/cl_ext.h> header file. */
/* #undef HAVE_OPENCL_CL_EXT_H */

/* Define is the OSX thread affinity policy macros defined */
/* #undef HAVE_OSX_THREAD_AFFINITY */

/* Define to 1 if you have the <picl.h> header file. */
/* #undef HAVE_PICL_H */

/* Define to 1 if you have the <poll.h> header file. */
/* #undef HAVE_POLL_H */

/* Define to 1 if you have the `posix_memalign' function. */
#define HAVE_POSIX_MEMALIGN 1

/* Cray style weak pragma */
/* #undef HAVE_PRAGMA_CRI_DUP */

/* HP style weak pragma */
/* #undef HAVE_PRAGMA_HP_SEC_DEF */

/* Supports weak pragma */
#define HAVE_PRAGMA_WEAK 1

/* Define to 1 if the system has the type `PROCESSOR_CACHE_TYPE'. */
/* #undef HAVE_PROCESSOR_CACHE_TYPE */

/* Define to 1 if the system has the type `PROCESSOR_GROUP_INFO'. */
/* #undef HAVE_PROCESSOR_GROUP_INFO */

/* Define to 1 if the system has the type `PROCESSOR_NUMBER'. */
/* #undef HAVE_PROCESSOR_NUMBER */

/* Define to 1 if the system has the type `PROCESSOR_RELATIONSHIP'. */
/* #undef HAVE_PROCESSOR_RELATIONSHIP */

/* Define to '1' if program_invocation_name is present and usable */
#define HAVE_PROGRAM_INVOCATION_NAME 1

/* Define to 1 if the system has the type `PSAPI_WORKING_SET_EX_BLOCK'. */
/* #undef HAVE_PSAPI_WORKING_SET_EX_BLOCK */

/* Define to 1 if the system has the type `PSAPI_WORKING_SET_EX_INFORMATION'.
   */
/* #undef HAVE_PSAPI_WORKING_SET_EX_INFORMATION */

/* Define to 1 if you have the <pthread_np.h> header file. */
/* #undef HAVE_PTHREAD_NP_H */

/* Define to 1 if the system has the type `pthread_t'. */
#define HAVE_PTHREAD_T 1

/* Define to 1 if you have the `ptrace' function. */
/* #undef HAVE_PTRACE */

/* Define if ptrace parameters available */
/* #undef HAVE_PTRACE_CONT */

/* Define to 1 if you have the `putenv' function. */
#define HAVE_PUTENV 1

/* Define to 1 if you have the `qsort' function. */
#define HAVE_QSORT 1

/* Define to 1 if you have the `rand' function. */
#define HAVE_RAND 1

/* Define to 1 if you have the <random.h> header file. */
/* #undef HAVE_RANDOM_H */

/* Define to 1 if you have the `random_r' function. */
/* #undef HAVE_RANDOM_R */

/* Define to 1 if the system has the type `RelationProcessorPackage'. */
/* #undef HAVE_RELATIONPROCESSORPACKAGE */

/* Define if ROMIO is enabled */
#define HAVE_ROMIO 1

/* Define to 1 if you have the `sched_getaffinity' function. */
#define HAVE_SCHED_GETAFFINITY 1

/* Define to 1 if you have the <sched.h> header file. */
#define HAVE_SCHED_H 1

/* Define to 1 if you have the `sched_setaffinity' function. */
#define HAVE_SCHED_SETAFFINITY 1

/* Define to 1 if you have the `select' function. */
/* #undef HAVE_SELECT */

/* Define to 1 if you have the `setitimer' function. */
#define HAVE_SETITIMER 1

/* Define to 1 if you have the `setlocale' function. */
#define HAVE_SETLOCALE 1

/* Define to 1 if you have the `setsid' function. */
/* #undef HAVE_SETSID */

/* Define to 1 if you have the `sigaction' function. */
/* #undef HAVE_SIGACTION */

/* Define to 1 if you have the `signal' function. */
#define HAVE_SIGNAL 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the `sigset' function. */
/* #undef HAVE_SIGSET */

/* Define to 1 if you have the `snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define if socklen_t is available */
/* #undef HAVE_SOCKLEN_T */

/* Define to 1 if you have the `srand' function. */
#define HAVE_SRAND 1

/* Define to 1 if the system has the type `ssize_t'. */
#define HAVE_SSIZE_T 1

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

/* Define to 1 if you have the `strftime' function. */
#define HAVE_STRFTIME 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strncasecmp' function. */
#define HAVE_STRNCASECMP 1

/* Define to 1 if you have the `strsignal' function. */
/* #undef HAVE_STRSIGNAL */

/* Define to 1 if you have the `strtoull' function. */
/* #undef HAVE_STRTOULL */

/* Define if struct ifconf can be used */
#define HAVE_STRUCT_IFCONF 1

/* Define if struct ifreq can be used */
#define HAVE_STRUCT_IFREQ 1

/* Define to 1 if the system has the type `struct random_data'. */
/* #undef HAVE_STRUCT_RANDOM_DATA */

/* Define to '1' if sysctl is present and usable */
/* #undef HAVE_SYSCTL */

/* Define to '1' if sysctlbyname is present and usable */
/* #undef HAVE_SYSCTLBYNAME */

/* Define to 1 if the system has the type
   `SYSTEM_LOGICAL_PROCESSOR_INFORMATION'. */
/* #undef HAVE_SYSTEM_LOGICAL_PROCESSOR_INFORMATION */

/* Define to 1 if the system has the type
   `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX'. */
/* #undef HAVE_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX */

/* Define if sys/bitypes.h exists */
#define HAVE_SYS_BITYPES_H 1

/* Define to 1 if you have the <sys/cpuset.h> header file. */
/* #undef HAVE_SYS_CPUSET_H */

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/ipc.h> header file. */
#define HAVE_SYS_IPC_H 1

/* Define to 1 if you have the <sys/lgrp_user.h> header file. */
/* #undef HAVE_SYS_LGRP_USER_H */

/* Define to 1 if you have the <sys/mman.h> header file. */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/poll.h> header file. */
/* #undef HAVE_SYS_POLL_H */

/* Define to 1 if you have the <sys/ptrace.h> header file. */
/* #undef HAVE_SYS_PTRACE_H */

/* Define to 1 if you have the <sys/select.h> header file. */
/* #undef HAVE_SYS_SELECT_H */

/* Define to 1 if you have the <sys/shm.h> header file. */
#define HAVE_SYS_SHM_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
/* #undef HAVE_SYS_SOCKIO_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/sysctl.h> header file. */
#define HAVE_SYS_SYSCTL_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H 1

/* Define to 1 if you have the <sys/utsname.h> header file. */
#define HAVE_SYS_UTSNAME_H 1

/* Define to enable tag error bits */
#define HAVE_TAG_ERROR_BITS 1

/* Define to 1 if you have the `thread_policy_set' function. */
/* #undef HAVE_THREAD_POLICY_SET */

/* Define to 1 if you have the `time' function. */
#define HAVE_TIME 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* define to enable timing collection */
/* #undef HAVE_TIMING */

/* Define if uint16_t is supported by the C compiler */
#define HAVE_UINT16_T 1

/* Define if uint32_t is supported by the C compiler */
#define HAVE_UINT32_T 1

/* Define if uint64_t is supported by the C compiler */
#define HAVE_UINT64_T 1

/* Define if uint8_t is supported by the C compiler */
#define HAVE_UINT8_T 1

/* Define to 1 if you have the `uname' function. */
#define HAVE_UNAME 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `unsetenv' function. */
/* #undef HAVE_UNSETENV */

/* Define to 1 if you have the `usleep' function. */
/* #undef HAVE_USLEEP */

/* Define to 1 if you have the `uuid_generate' function. */
/* #undef HAVE_UUID_GENERATE */

/* Define to 1 if you have the <uuid/uuid.h> header file. */
#define HAVE_UUID_UUID_H 1

/* Define to 1 if you have the <valgrind/valgrind.h> header file. */
/* #undef HAVE_VALGRIND_VALGRIND_H */

/* Define if we have va_copy */
#define HAVE_VA_COPY 1

/* Whether C compiler supports symbol visibility or not */
#define HAVE_VISIBILITY 1

/* Define to 1 if you have the `vsnprintf' function. */
#define HAVE_VSNPRINTF 1

/* Define to 1 if you have the `vsprintf' function. */
#define HAVE_VSPRINTF 1

/* Define to 1 if you have the <wait.h> header file. */
/* #undef HAVE_WAIT_H */

/* Attribute style weak pragma */
#define HAVE_WEAK_ATTRIBUTE 1

/* Define to 1 if you have the <X11/keysym.h> header file. */
/* #undef HAVE_X11_KEYSYM_H */

/* Define to 1 if you have the <X11/Xlib.h> header file. */
/* #undef HAVE_X11_XLIB_H */

/* Define to 1 if you have the <X11/Xutil.h> header file. */
/* #undef HAVE_X11_XUTIL_H */

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* define if the compiler defines __FUNCTION__ */
#define HAVE__FUNCTION__ /**/

/* define if the compiler defines __func__ */
#define HAVE__FUNC__ /**/

/* Define to '1' if __progname is present and usable */
#define HAVE___PROGNAME 1

/* Define if we have __va_copy */
/* #undef HAVE___VA_COPY */

/* Define to 1 on AIX */
/* #undef HWLOC_AIX_SYS */

/* Define to 1 on BlueGene/Q */
/* #undef HWLOC_BGQ_SYS */

/* Whether C compiler supports symbol visibility or not */
#define HWLOC_C_HAVE_VISIBILITY 0

/* Define to 1 on Darwin */
/* #undef HWLOC_DARWIN_SYS */

/* Whether we are in debugging mode or not */
/* #undef HWLOC_DEBUG */

/* Define to 1 on *FREEBSD */
/* #undef HWLOC_FREEBSD_SYS */

/* Whether your compiler has __attribute__ or not */
#define HWLOC_HAVE_ATTRIBUTE 1

/* Whether your compiler has __attribute__ aligned or not */
#define HWLOC_HAVE_ATTRIBUTE_ALIGNED 1

/* Whether your compiler has __attribute__ always_inline or not */
#define HWLOC_HAVE_ATTRIBUTE_ALWAYS_INLINE 1

/* Whether your compiler has __attribute__ cold or not */
#define HWLOC_HAVE_ATTRIBUTE_COLD 1

/* Whether your compiler has __attribute__ const or not */
#define HWLOC_HAVE_ATTRIBUTE_CONST 1

/* Whether your compiler has __attribute__ deprecated or not */
#define HWLOC_HAVE_ATTRIBUTE_DEPRECATED 1

/* Whether your compiler has __attribute__ format or not */
#define HWLOC_HAVE_ATTRIBUTE_FORMAT 1

/* Whether your compiler has __attribute__ hot or not */
#define HWLOC_HAVE_ATTRIBUTE_HOT 1

/* Whether your compiler has __attribute__ malloc or not */
#define HWLOC_HAVE_ATTRIBUTE_MALLOC 1

/* Whether your compiler has __attribute__ may_alias or not */
#define HWLOC_HAVE_ATTRIBUTE_MAY_ALIAS 1

/* Whether your compiler has __attribute__ nonnull or not */
#define HWLOC_HAVE_ATTRIBUTE_NONNULL 1

/* Whether your compiler has __attribute__ noreturn or not */
#define HWLOC_HAVE_ATTRIBUTE_NORETURN 1

/* Whether your compiler has __attribute__ no_instrument_function or not */
#define HWLOC_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION 1

/* Whether your compiler has __attribute__ packed or not */
#define HWLOC_HAVE_ATTRIBUTE_PACKED 1

/* Whether your compiler has __attribute__ pure or not */
#define HWLOC_HAVE_ATTRIBUTE_PURE 1

/* Whether your compiler has __attribute__ sentinel or not */
#define HWLOC_HAVE_ATTRIBUTE_SENTINEL 1

/* Whether your compiler has __attribute__ unused or not */
#define HWLOC_HAVE_ATTRIBUTE_UNUSED 1

/* Whether your compiler has __attribute__ warn unused result or not */
#define HWLOC_HAVE_ATTRIBUTE_WARN_UNUSED_RESULT 1

/* Whether your compiler has __attribute__ weak alias or not */
#define HWLOC_HAVE_ATTRIBUTE_WEAK_ALIAS 1

/* Define to 1 if your `ffs' function is known to be broken. */
/* #undef HWLOC_HAVE_BROKEN_FFS */

/* Define to 1 if you have the `clz' function. */
/* #undef HWLOC_HAVE_CLZ */

/* Define to 1 if you have the `clzl' function. */
/* #undef HWLOC_HAVE_CLZL */

/* Define to 1 if the CPU_SET macro works */
#define HWLOC_HAVE_CPU_SET 1

/* Define to 1 if the CPU_SET_S macro works */
#define HWLOC_HAVE_CPU_SET_S 1

/* Define to 1 if you have the `cudart' SDK. */
/* #undef HWLOC_HAVE_CUDART */

/* Define to 1 if function `clz' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_CLZ */

/* Define to 1 if function `clzl' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_CLZL */

/* Define to 1 if function `ffs' is declared by system headers */
#define HWLOC_HAVE_DECL_FFS 1

/* Define to 1 if function `ffsl' is declared by system headers */
#define HWLOC_HAVE_DECL_FFSL 1

/* Define to 1 if function `fls' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FLS */

/* Define to 1 if function `flsl' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FLSL */

/* Define to 1 if function `strncasecmp' is declared by system headers */
#define HWLOC_HAVE_DECL_STRNCASECMP 1

/* Define to 1 if you have the `ffs' function. */
#define HWLOC_HAVE_FFS 1

/* Define to 1 if you have the `ffsl' function. */
#define HWLOC_HAVE_FFSL 1

/* Define to 1 if you have the `fls' function. */
/* #undef HWLOC_HAVE_FLS */

/* Define to 1 if you have the `flsl' function. */
/* #undef HWLOC_HAVE_FLSL */

/* Define to 1 if you have the GL module components. */
/* #undef HWLOC_HAVE_GL */

/* Define to 1 if you have libudev. */
/* #undef HWLOC_HAVE_LIBUDEV */

/* Define to 1 if you have the `libxml2' library. */
/* #undef HWLOC_HAVE_LIBXML2 */

/* Define to 1 if building the Linux I/O component */
#define HWLOC_HAVE_LINUXIO 1

/* Define to 1 if enabling Linux-specific PCI discovery in the Linux I/O
   component */
#define HWLOC_HAVE_LINUXPCI 1

/* Define to 1 if you have the `NVML' library. */
/* #undef HWLOC_HAVE_NVML */

/* Define to 1 if glibc provides the old prototype (without length) of
   sched_setaffinity() */
/* #undef HWLOC_HAVE_OLD_SCHED_SETAFFINITY */

/* Define to 1 if you have the `OpenCL' library. */
/* #undef HWLOC_HAVE_OPENCL */

/* Define to 1 if the hwloc library should support dynamically-loaded plugins
   */
/* #undef HWLOC_HAVE_PLUGINS */

/* `Define to 1 if you have pthread_getthrds_np' */
/* #undef HWLOC_HAVE_PTHREAD_GETTHRDS_NP */

/* Define to 1 if pthread mutexes are available */
#define HWLOC_HAVE_PTHREAD_MUTEX 1

/* Define to 1 if glibc provides a prototype of sched_setaffinity() */
#define HWLOC_HAVE_SCHED_SETAFFINITY 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HWLOC_HAVE_STDINT_H 1

/* Define to 1 if function `syscall' is available with 6 parameters */
#define HWLOC_HAVE_SYSCALL 1

/* Define to 1 if you have the `windows.h' header. */
/* #undef HWLOC_HAVE_WINDOWS_H */

/* Define to 1 if X11 headers including Xutil.h and keysym.h are available. */
/* #undef HWLOC_HAVE_X11_KEYSYM */

/* Define to 1 if you have x86 cpuid */
#define HWLOC_HAVE_X86_CPUID 1

/* Define to 1 on HP-UX */
/* #undef HWLOC_HPUX_SYS */

/* Define to 1 on Irix */
/* #undef HWLOC_IRIX_SYS */

/* Define to 1 on Linux */
#define HWLOC_LINUX_SYS 1

/* Define to 1 on *NETBSD */
/* #undef HWLOC_NETBSD_SYS */

/* The size of `unsigned int', as computed by sizeof */
#define HWLOC_SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof */
#define HWLOC_SIZEOF_UNSIGNED_LONG 8

/* Define to 1 on Solaris */
/* #undef HWLOC_SOLARIS_SYS */

/* The hwloc symbol prefix */
#define HWLOC_SYM_PREFIX hwloc_

/* The hwloc symbol prefix in all caps */
#define HWLOC_SYM_PREFIX_CAPS HWLOC_

/* Whether we need to re-define all the hwloc public symbols or not */
#define HWLOC_SYM_TRANSFORM 0

/* Define to 1 on unsupported systems */
/* #undef HWLOC_UNSUPPORTED_SYS */

/* The library version, always available, even in embedded mode, contrary to
   VERSION */
#define HWLOC_VERSION "2.0.1rc2-git"

/* Define to 1 on WINDOWS */
/* #undef HWLOC_WIN_SYS */

/* Define to 1 on x86_32 */
/* #undef HWLOC_X86_32_ARCH */

/* Define to 1 on x86_64 */
#define HWLOC_X86_64_ARCH 1

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
#define MPICH_ERROR_MSG_LEVEL MPICH_ERROR_MSG__ALL

/* Define as the name of the debugger support library */
/* #undef MPICH_INFODLL_LOC */

/* MPICH is configured to require thread safety */
#define MPICH_IS_THREADED 1

/* Method used to implement atomic updates and access */
#define MPICH_THREAD_GRANULARITY MPICH_THREAD_GRANULARITY__GLOBAL

/* Level of thread support selected at compile time */
#define MPICH_THREAD_LEVEL MPI_THREAD_MULTIPLE

/* Method used to implement refcount updates */
#define MPICH_THREAD_REFCOUNT MPICH_REFCOUNT__NONE

/* define to disable reference counting predefined objects like MPI_COMM_WORLD
   */
/* #undef MPICH_THREAD_SUPPRESS_PREDEFINED_REFCOUNTS */

/* Define to enable message-driven thread activation */
/* #undef MPICH_THREAD_USE_MDTA */

/* CH4 should build locality info */
/* #undef MPIDI_BUILD_CH4_LOCALITY_INFO */

/* Define if CH4U will use per-communicator message queues */
/* #undef MPIDI_CH4U_USE_PER_COMM_QUEUE */

/* CH4 Directly transfers data through the chosen netmode */
/* #undef MPIDI_CH4_DIRECT_NETMOD */

/* Define to use bgq capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_BGQ */

/* Define to use gni capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_GNI */

/* Define to use PSM capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_PSM */

/* Define to use PSM2 capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_PSM2 */

/* Define to use runtime capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_RUNTIME */

/* Define to use sockets capability set */
/* #undef MPIDI_CH4_OFI_USE_SET_SOCKETS */

/* Define to enable direct multi-threading model */
/* #undef MPIDI_CH4_USE_MT_DIRECT */

/* Define to enable hand-off multi-threading model */
/* #undef MPIDI_CH4_USE_MT_HANDOFF */

/* Define to enable runtime multi-threading model */
/* #undef MPIDI_CH4_USE_MT_RUNTIME */

/* Define to enable trylock-enqueue multi-threading model */
/* #undef MPIDI_CH4_USE_MT_TRYLOCK */

/* Define to turn on the inlining optimizations in Nemesis code */
#define MPID_NEM_INLINE 1

/* Method for local large message transfers. */
#define MPID_NEM_LOCAL_LMT_IMPL MPID_NEM_LOCAL_LMT_SHM_COPY

/* Define to enable lock-free communication queues */
#define MPID_NEM_USE_LOCK_FREE_QUEUES 1

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

/* Define to enable timing mutexes */
/* #undef MPIU_MUTEX_WAIT_TIME */

/* Define if /bin must be in path */
/* #undef NEEDS_BIN_IN_PATH */

/* Define if environ decl needed */
/* #undef NEEDS_ENVIRON_DECL */

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

/* Define if vsnprintf needs a declaration */
/* #undef NEEDS_VSNPRINTF_DECL */

/* Name of package */
#define PACKAGE "mpich"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "discuss@mpich.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "MPICH"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "MPICH 3.3rc1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "mpich"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://www.mpich.org/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.3rc1"

/* Define to turn on the prefetching optimization in Nemesis code */
#define PREFETCH_CELL 1

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

/* The size of `MPII_Bsend_data_t', as computed by sizeof. */
#define SIZEOF_MPII_BSEND_DATA_T 96

/* The size of `OPA_ptr_t', as computed by sizeof. */
#define SIZEOF_OPA_PTR_T 8

/* define if sizeof(void *) = sizeof(MPI_Aint) */
#define SIZEOF_PTR_IS_AINT 1

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
#define STRERROR_R_CHAR_P 1

/* Define TRUE */
#define TRUE 1

/* Define if MPI_Aint should be used instead of void * for storing attribute
   values */
/* #undef USE_AINT_FOR_ATTRVAL */

/* Define if alloca should be used if available */
/* #undef USE_ALLOCA */

/* Define if performing coverage tests */
/* #undef USE_COVERAGE */

/* Define to use the fastboxes in Nemesis code */
#define USE_FASTBOX 1

/* Define if file should be used for name publisher */
/* #undef USE_FILE_FOR_NAMEPUB */

/* Define if the length of a CHARACTER*(*) string in Fortran should be passed
   as size_t instead of int */
/* #undef USE_FORT_STR_LEN_SIZET */

/* Enable extensions on HP-UX. */
#ifndef _HPUX_SOURCE
# define _HPUX_SOURCE 1
#endif


/* define to choose logging library */
#define USE_LOGGING MPICH_LOGGING__NONE

/* Define to enable memory tracing */
/* #undef USE_MEMORY_TRACING */

/* Define if mpiexec should create a new process group session */
/* #undef USE_NEW_SESSION */

/* Define if _POSIX_C_SOURCE needs to be undefined for struct ifconf */
/* #undef USE_NOPOSIX_FOR_IFCONF */

/* Define if PMI2 API must be used */
/* #undef USE_PMI2_API */

/* Define if PMIx API must be used */
/* #undef USE_PMIX_API */

/* Define if access to PMI information through a port rather than just an fd
   is allowed */
#define USE_PMI_PORT 1

/* Define if sigaction should be used to set signals */
/* #undef USE_SIGACTION */

/* Define if signal should be used to set signals */
/* #undef USE_SIGNAL */

/* Define it the socket verify macros should be enabled */
/* #undef USE_SOCK_VERIFY */

/* Define if _SVID_SOURCE needs to be defined for struct ifconf */
/* #undef USE_SVIDSOURCE_FOR_IFCONF */

/* Define if we can use a symmetric heap */
/* #undef USE_SYM_HEAP */

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif


/* Define if weak symbols should be used */
#define USE_WEAK_SYMBOLS 1

/* Version number of package */
#define VERSION "3.3rc1"

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

/* Define to 1 if the X Window System is missing or not being used. */
#define X_DISPLAY_MISSING 1

/* Are we building for HP-UX? */
#define _HPUX_SOURCE 1

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */

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

/* define if bool is a built-in type */
/* #undef bool */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define this to the process ID type */
#define hwloc_pid_t pid_t

/* Define this to the thread ID type */
#define hwloc_thread_t pthread_t

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

#endif /* !defined(MPICHCONF_H_INCLUDED) */

