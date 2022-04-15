#ifndef _INCLUDE_MPLCONFIG_H
#define _INCLUDE_MPLCONFIG_H 1
 
/* include/mplconfig.h. Generated automatically at end of configure. */
/* include/config.h.  Generated from config.h.in by configure.  */
/* include/config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to 1 if MPL enables MPL_aligned_alloc. */
#ifndef MPL_DEFINE_ALIGNED_ALLOC 
#define MPL_DEFINE_ALIGNED_ALLOC  1 
#endif

/* Define to 1 if you have the `abt_cleanup_push' function. */
/* #undef HAVE_ABT_CLEANUP_PUSH */

/* Define to 1 if you have the <abt.h> header file. */
/* #undef HAVE_ABT_H */

/* Define to 1 if you have the `ABT_thread_yield' function. */
/* #undef HAVE_ABT_THREAD_YIELD */

/* Define to 1 if you have the `aligned_alloc' function. */
#ifndef MPL_HAVE_ALIGNED_ALLOC 
#define MPL_HAVE_ALIGNED_ALLOC  1 
#endif

/* Define to 1 if you have the <arpa/inet.h> header file. */
#ifndef MPL_HAVE_ARPA_INET_H 
#define MPL_HAVE_ARPA_INET_H  1 
#endif

/* Define to 1 if you have the `backtrace_symbols' function. */
#ifndef MPL_HAVE_BACKTRACE_SYMBOLS 
#define MPL_HAVE_BACKTRACE_SYMBOLS  1 
#endif

/* define if valgrind is old and/or broken compared to what we are expecting
   */
#ifndef MPL_HAVE_BROKEN_VALGRIND 
#define MPL_HAVE_BROKEN_VALGRIND  1 
#endif

/* Define to 1 if the compiler supports __builtin_expect. */
#ifndef MPL_HAVE_BUILTIN_EXPECT 
#define MPL_HAVE_BUILTIN_EXPECT  1 
#endif

/* Define to 1 if you have the `clock_getres' function. */
#ifndef MPL_HAVE_CLOCK_GETRES 
#define MPL_HAVE_CLOCK_GETRES  1 
#endif

/* Define to 1 if you have the `clock_gettime' function. */
#ifndef MPL_HAVE_CLOCK_GETTIME 
#define MPL_HAVE_CLOCK_GETTIME  1 
#endif

/* Define to 1 if you have the <ctype.h> header file. */
#ifndef MPL_HAVE_CTYPE_H 
#define MPL_HAVE_CTYPE_H  1 
#endif

/* Define to 1 if you have the declaration of `backtrace_create_state', and to
   0 if you don't. */
#ifndef MPL_HAVE_DECL_BACKTRACE_CREATE_STATE 
#define MPL_HAVE_DECL_BACKTRACE_CREATE_STATE  0 
#endif

/* Define to 1 if you have the declaration of `backtrace_print', and to 0 if
   you don't. */
#ifndef MPL_HAVE_DECL_BACKTRACE_PRINT 
#define MPL_HAVE_DECL_BACKTRACE_PRINT  0 
#endif

/* Define to 1 if you have the <dlfcn.h> header file. */
#ifndef MPL_HAVE_DLFCN_H 
#define MPL_HAVE_DLFCN_H  1 
#endif

/* Define to 1 if you have the <drd.h> header file. */
/* #undef HAVE_DRD_H */

/* Define to 1 if you have the <errno.h> header file. */
#ifndef MPL_HAVE_ERRNO_H 
#define MPL_HAVE_ERRNO_H  1 
#endif

/* Define to 1 if you have the <execinfo.h> header file. */
#ifndef MPL_HAVE_EXECINFO_H 
#define MPL_HAVE_EXECINFO_H  1 
#endif

/* Define to 1 if you have the `fdopen' function. */
#ifndef MPL_HAVE_FDOPEN 
#define MPL_HAVE_FDOPEN  1 
#endif

/* Define to 1 if the system has the `fallthrough' function attribute */
/* #undef HAVE_FUNC_ATTRIBUTE_FALLTHROUGH */

/* Define if GNU __attribute__ is supported */
#ifndef MPL_HAVE_GCC_ATTRIBUTE 
#define MPL_HAVE_GCC_ATTRIBUTE  1 
#endif

/* Define to 1 if you have the `gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if you have the `getifaddrs' function. */
#ifndef MPL_HAVE_GETIFADDRS 
#define MPL_HAVE_GETIFADDRS  1 
#endif

/* Define to 1 if you have the `getpid' function. */
#ifndef MPL_HAVE_GETPID 
#define MPL_HAVE_GETPID  1 
#endif

/* Define to 1 if you have the `gettimeofday' function. */
#ifndef MPL_HAVE_GETTIMEOFDAY 
#define MPL_HAVE_GETTIMEOFDAY  1 
#endif

/* Define to 1 if you have the <helgrind.h> header file. */
/* #undef HAVE_HELGRIND_H */

/* Define to 1 if you have the <ifaddrs.h> header file. */
#ifndef MPL_HAVE_IFADDRS_H 
#define MPL_HAVE_IFADDRS_H  1 
#endif

/* Define to 1 if you have the `inet_ntop' function. */
#ifndef MPL_HAVE_INET_NTOP 
#define MPL_HAVE_INET_NTOP  1 
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef MPL_HAVE_INTTYPES_H 
#define MPL_HAVE_INTTYPES_H  1 
#endif

/* Define to 1 if you have the backtrace header (backtrace.h) and library
   (-lbacktrace) */
/* #undef HAVE_LIBBACKTRACE */

/* Define to 1 if you have the libunwind header (libunwind.h) and library
   (-lunwind) */
/* #undef HAVE_LIBUNWIND */

/* Define to 1 if you have the `uti' library (-luti). */
/* #undef HAVE_LIBUTI */

/* Define to 1 if you have the `mach_absolute_time' function. */
/* #undef HAVE_MACH_ABSOLUTE_TIME */

/* Define if C99-style variable argument list macro functionality */
#ifndef MPL_HAVE_MACRO_VA_ARGS 
#define MPL_HAVE_MACRO_VA_ARGS  1 
#endif

/* Define to 1 if you have the <memcheck.h> header file. */
/* #undef HAVE_MEMCHECK_H */

/* Define to 1 if you have the <memory.h> header file. */
#ifndef MPL_HAVE_MEMORY_H 
#define MPL_HAVE_MEMORY_H  1 
#endif

/* Define to 1 if you have the `mkstemp' function. */
#ifndef MPL_HAVE_MKSTEMP 
#define MPL_HAVE_MKSTEMP  1 
#endif

/* Define to 1 if you have the `mmap' function. */
#ifndef MPL_HAVE_MMAP 
#define MPL_HAVE_MMAP  1 
#endif

/* Define to 1 if you have the `munmap' function. */
#ifndef MPL_HAVE_MUNMAP 
#define MPL_HAVE_MUNMAP  1 
#endif

/* Define to 1 if you have the `posix_memalign' function. */
#ifndef MPL_HAVE_POSIX_MEMALIGN 
#define MPL_HAVE_POSIX_MEMALIGN  1 
#endif

/* Define to 1 if you have the `pthread_cleanup_push' function. */
/* #undef HAVE_PTHREAD_CLEANUP_PUSH */

/* Define if pthread_cleanup_push is available, even as a macro */
/* #undef HAVE_PTHREAD_CLEANUP_PUSH_MACRO */

/* Define to 1 if you have the <pthread.h> header file. */
#ifndef MPL_HAVE_PTHREAD_H 
#define MPL_HAVE_PTHREAD_H  1 
#endif

/* Define if pthread_mutexattr_setpshared is available. */
#ifndef MPL_HAVE_PTHREAD_MUTEXATTR_SETPSHARED 
#define MPL_HAVE_PTHREAD_MUTEXATTR_SETPSHARED  1 
#endif

/* Define to 1 if you have the `pthread_yield' function. */
#ifndef MPL_HAVE_PTHREAD_YIELD 
#define MPL_HAVE_PTHREAD_YIELD  1 
#endif

/* Define to 1 if you have the `putenv' function. */
#ifndef MPL_HAVE_PUTENV 
#define MPL_HAVE_PUTENV  1 
#endif

/* Define to 1 if you have the <sched.h> header file. */
#ifndef MPL_HAVE_SCHED_H 
#define MPL_HAVE_SCHED_H  1 
#endif

/* Define to 1 if you have the `sched_yield' function. */
#ifndef MPL_HAVE_SCHED_YIELD 
#define MPL_HAVE_SCHED_YIELD  1 
#endif

/* Define to 1 if you have the `select' function. */
#ifndef MPL_HAVE_SELECT 
#define MPL_HAVE_SELECT  1 
#endif

/* Define to 1 if you have the `shmat' function. */
/* #undef HAVE_SHMAT */

/* Define to 1 if you have the `shmctl' function. */
/* #undef HAVE_SHMCTL */

/* Define to 1 if you have the `shmdt' function. */
/* #undef HAVE_SHMDT */

/* Define to 1 if you have the `shmget' function. */
/* #undef HAVE_SHMGET */

/* Define to 1 if you have the `sleep' function. */
#ifndef MPL_HAVE_SLEEP 
#define MPL_HAVE_SLEEP  1 
#endif

/* Define to 1 if you have the `snprintf' function. */
#ifndef MPL_HAVE_SNPRINTF 
#define MPL_HAVE_SNPRINTF  1 
#endif

/* Define to 1 if you have the <stdarg.h> header file. */
#ifndef MPL_HAVE_STDARG_H 
#define MPL_HAVE_STDARG_H  1 
#endif

/* Define to 1 if stdbool.h conforms to C99. */
#ifndef MPL_HAVE_STDBOOL_H 
#define MPL_HAVE_STDBOOL_H  1 
#endif

/* Define to 1 if you have the <stdint.h> header file. */
#ifndef MPL_HAVE_STDINT_H 
#define MPL_HAVE_STDINT_H  1 
#endif

/* Define to 1 if you have the <stdio.h> header file. */
#ifndef MPL_HAVE_STDIO_H 
#define MPL_HAVE_STDIO_H  1 
#endif

/* Define to 1 if you have the <stdlib.h> header file. */
#ifndef MPL_HAVE_STDLIB_H 
#define MPL_HAVE_STDLIB_H  1 
#endif

/* Define to 1 if you have the `strdup' function. */
#ifndef MPL_HAVE_STRDUP 
#define MPL_HAVE_STRDUP  1 
#endif

/* Define to 1 if you have the `strerror' function. */
#ifndef MPL_HAVE_STRERROR 
#define MPL_HAVE_STRERROR  1 
#endif

/* Define to 1 if you have the <strings.h> header file. */
#ifndef MPL_HAVE_STRINGS_H 
#define MPL_HAVE_STRINGS_H  1 
#endif

/* Define to 1 if you have the <string.h> header file. */
#ifndef MPL_HAVE_STRING_H 
#define MPL_HAVE_STRING_H  1 
#endif

/* Define to 1 if you have the `strncmp' function. */
#ifndef MPL_HAVE_STRNCMP 
#define MPL_HAVE_STRNCMP  1 
#endif

/* Define to 1 if you have the <sys/mman.h> header file. */
#ifndef MPL_HAVE_SYS_MMAN_H 
#define MPL_HAVE_SYS_MMAN_H  1 
#endif

/* Define to 1 if you have the <sys/select.h> header file. */
#ifndef MPL_HAVE_SYS_SELECT_H 
#define MPL_HAVE_SYS_SELECT_H  1 
#endif

/* Define to 1 if you have the <sys/stat.h> header file. */
#ifndef MPL_HAVE_SYS_STAT_H 
#define MPL_HAVE_SYS_STAT_H  1 
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#ifndef MPL_HAVE_SYS_TYPES_H 
#define MPL_HAVE_SYS_TYPES_H  1 
#endif

/* Define to 1 if you have the <sys/uio.h> header file. */
#ifndef MPL_HAVE_SYS_UIO_H 
#define MPL_HAVE_SYS_UIO_H  1 
#endif

/* Define to 1 if you have the <thread.h> header file. */
/* #undef HAVE_THREAD_H */

/* Define to 1 if you have the `thr_yield' function. */
/* #undef HAVE_THR_YIELD */

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef MPL_HAVE_UNISTD_H 
#define MPL_HAVE_UNISTD_H  1 
#endif

/* Define to 1 if you have the `usleep' function. */
#ifndef MPL_HAVE_USLEEP 
#define MPL_HAVE_USLEEP  1 
#endif

/* Define to 1 if you have the <valgrind/drd.h> header file. */
/* #undef HAVE_VALGRIND_DRD_H */

/* Define to 1 if you have the <valgrind.h> header file. */
/* #undef HAVE_VALGRIND_H */

/* Define to 1 if you have the <valgrind/helgrind.h> header file. */
/* #undef HAVE_VALGRIND_HELGRIND_H */

/* Define to 1 if you have the <valgrind/memcheck.h> header file. */
/* #undef HAVE_VALGRIND_MEMCHECK_H */

/* Define to 1 if you have the <valgrind/valgrind.h> header file. */
/* #undef HAVE_VALGRIND_VALGRIND_H */

/* Define to 1 if the system has the `aligned' variable attribute */
#ifndef MPL_HAVE_VAR_ATTRIBUTE_ALIGNED 
#define MPL_HAVE_VAR_ATTRIBUTE_ALIGNED  1 
#endif

/* Define to 1 if the system has the `used' variable attribute */
#ifndef MPL_HAVE_VAR_ATTRIBUTE_USED 
#define MPL_HAVE_VAR_ATTRIBUTE_USED  1 
#endif

/* Define to 1 if you have the <windows.h> header file. */
/* #undef HAVE_WINDOWS_H */

/* Define to 1 if you have the `yield' function. */
/* #undef HAVE_YIELD */

/* Define to 1 if the system has the type `_Bool'. */
#ifndef MPL_HAVE__BOOL 
#define MPL_HAVE__BOOL  1 
#endif

/* defined if the C compiler supports __typeof(variable) */
#ifndef MPL_HAVE___TYPEOF 
#define MPL_HAVE___TYPEOF  1 
#endif

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_CPUID_RDTSC32 */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_CPUID_RDTSC64 */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_RDTSC */

/* Define which x86 cycle counter to use */
/* #undef LINUX86_CYCLE_RDTSCP */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#ifndef MPL_LT_OBJDIR 
#define MPL_LT_OBJDIR  ".libs/" 
#endif

/* Define if use MMAP shared memory */
#ifndef MPL_MPL_USE_MMAP_SHM 
#define MPL_MPL_USE_MMAP_SHM  1 
#endif

/* Define if use Windows shared memory */
/* #undef MPL_USE_NT_SHM */

/* Define if use SYSV shared memory */
/* #undef MPL_USE_SYSV_SHM */

/* Define if aligned_alloc needs a declaration */
/* #undef NEEDS_ALIGNED_ALLOC_DECL */

/* Define if fdopen needs a declaration */
#ifndef MPL_NEEDS_FDOPEN_DECL 
#define MPL_NEEDS_FDOPEN_DECL  1 
#endif

/* Define if mkstemp needs a declaration */
/* #undef NEEDS_MKSTEMP_DECL */

/* Define if pthread_mutexattr_settype needs a declaration */
/* #undef NEEDS_PTHREAD_MUTEXATTR_SETTYPE_DECL */

/* Define if putenv needs a declaration */
/* #undef NEEDS_PUTENV_DECL */

/* Define if snprintf needs a declaration */
/* #undef NEEDS_SNPRINTF_DECL */

/* Define if strdup needs a declaration */
/* #undef NEEDS_STRDUP_DECL */

/* Define if strerror needs a declaration */
/* #undef NEEDS_STRERROR_DECL */

/* Define if strncmp needs a declaration */
/* #undef NEEDS_STRNCMP_DECL */

/* Define if sys/time.h is required to get timer definitions */
/* #undef NEEDS_SYS_TIME_H */

/* Define if usleep needs a declaration */
/* #undef NEEDS_USLEEP_DECL */

/* Name of package */
#ifndef MPL_PACKAGE 
#define MPL_PACKAGE  "mpl" 
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef MPL_PACKAGE_BUGREPORT 
#define MPL_PACKAGE_BUGREPORT  "" 
#endif

/* Define to the full name of this package. */
#ifndef MPL_PACKAGE_NAME 
#define MPL_PACKAGE_NAME  "MPL" 
#endif

/* Define to the full name and version of this package. */
#ifndef MPL_PACKAGE_STRING 
#define MPL_PACKAGE_STRING  "MPL 0.1" 
#endif

/* Define to the one symbol short name of this package. */
#ifndef MPL_PACKAGE_TARNAME 
#define MPL_PACKAGE_TARNAME  "mpl" 
#endif

/* Define to the home page for this package. */
#ifndef MPL_PACKAGE_URL 
#define MPL_PACKAGE_URL  "" 
#endif

/* Define to the version of this package. */
#ifndef MPL_PACKAGE_VERSION 
#define MPL_PACKAGE_VERSION  "0.1" 
#endif

/* set to the name of the interprocess mutex package */
#ifndef MPL_PROC_MUTEX_PACKAGE_NAME 
#define MPL_PROC_MUTEX_PACKAGE_NAME  MPL_PROC_MUTEX_PACKAGE_POSIX 
#endif

/* Define to an expression that will result in an error checking mutex type.
   */
/* #undef PTHREAD_MUTEX_ERRORCHECK_VALUE */

/* Define to 1 if you have the ANSI C header files. */
#ifndef MPL_STDC_HEADERS 
#define MPL_STDC_HEADERS  1 
#endif

/* set to the name of the thread package */
#ifndef MPL_THREAD_PACKAGE_NAME 
#define MPL_THREAD_PACKAGE_NAME  MPL_THREAD_PACKAGE_POSIX 
#endif

/* If the compiler supports a TLS storage class define it to that here */
/* #undef TLS_SPECIFIER */

/* Define if performing coverage tests */
/* #undef USE_COVERAGE */

/* Define to enable logging macros */
/* #undef USE_DBG_LOGGING */

/* Define to enable memory tracing */
/* #undef USE_MEMORY_TRACING */

/* Define if we have sysv shared memory */
#ifndef MPL_USE_MMAP_SHM 
#define MPL_USE_MMAP_SHM  1 
#endif

/* Define to use nothing to yield processor */
#ifndef MPL_USE_NOTHING_FOR_YIELD 
#define MPL_USE_NOTHING_FOR_YIELD  1 
#endif

/* Define to use sched_yield to yield processor */
/* #undef USE_SCHED_YIELD_FOR_YIELD */

/* Define to use select to yield processor */
/* #undef USE_SELECT_FOR_YIELD */

/* Define to use sleep to yield processor */
/* #undef USE_SLEEP_FOR_YIELD */

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


/* Define if we have sysv shared memory */
/* #undef USE_SYSV_SHM */

/* Define to use usleep to yield processor */
/* #undef USE_USLEEP_FOR_YIELD */

/* Define to use yield to yield processor */
/* #undef USE_YIELD_FOR_YIELD */

/* Version number of package */
#ifndef MPL_VERSION 
#define MPL_VERSION  "0.1" 
#endif

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#ifndef _mpl_restrict 
#define _mpl_restrict  __restrict 
#endif
/* Work around a bug in Sun C++: it does not support _Restrict or
   __restrict__, even though the corresponding Sun C compiler ends up with
   "#define restrict _Restrict" or "#define restrict __restrict__" in the
   previous line.  Perhaps some future version of Sun C++ will work with
   restrict; if so, hopefully it defines __RESTRICT like Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
# define __restrict__
#endif
 
/* once: _INCLUDE_MPLCONFIG_H */
#endif
