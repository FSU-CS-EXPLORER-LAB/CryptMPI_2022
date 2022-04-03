/* src/config.h.  Generated from config.h.in by configure.  */
/* src/config.h.in.  Generated from configure.ac by autoheader.  */

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2008 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */


/* define if lock-based emulation was explicitly requested at configure time
   via --with-atomic-primitives=no */
/* #undef EXPLICIT_EMULATION */

/* Define to 1 if you have the <atomic.h> header file. */
/* #undef HAVE_ATOMIC_H */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* define to 1 if we have support for gcc ARM atomics */
/* #undef HAVE_GCC_AND_ARM_ASM */

/* define to 1 if we have support for gcc ia64 primitives */
/* #undef HAVE_GCC_AND_IA64_ASM */

/* define to 1 if we have support for gcc PowerPC atomics */
/* #undef HAVE_GCC_AND_POWERPC_ASM */

/* define to 1 if we have support for gcc SiCortex atomics */
/* #undef HAVE_GCC_AND_SICORTEX_ASM */

/* Define if GNU __attribute__ is supported */
#define HAVE_GCC_ATTRIBUTE 1

/* define to 1 if we have support for gcc atomic intrinsics */
#define HAVE_GCC_INTRINSIC_ATOMICS 1

/* define to 1 if we have support for gcc x86/x86_64 primitives */
#define HAVE_GCC_X86_32_64 1

/* define to 1 if we have support for gcc x86 primitives for pre-Pentium 4 */
#define HAVE_GCC_X86_32_64_P3 1

/* Define to 1 if you have the <intrin.h> header file. */
/* #undef HAVE_INTRIN_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `pthread' library (-lpthread). */
#define HAVE_LIBPTHREAD 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* define to 1 if we have support for Windows NT atomic intrinsics */
/* #undef HAVE_NT_INTRINSICS */

/* Define to 1 if you have the <pthread.h> header file. */
#define HAVE_PTHREAD_H 1

/* Define to 1 if you have the `pthread_yield' function. */
#define HAVE_PTHREAD_YIELD 1

/* Define to 1 if you have the `sched_yield' function. */
/* #undef HAVE_SCHED_YIELD */

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define if strict checking of atomic operation fairness is desired */
/* #undef HAVE_STRICT_FAIRNESS_CHECKS */

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* define to 1 if we have support for Sun atomic operations library */
/* #undef HAVE_SUN_ATOMIC_OPS */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* define to the maximum number of simultaneous threads */
#define MAX_NTHREADS 100

/* Define to 1 if assertions should be disabled. */
/* #undef NDEBUG */

/* Name of package */
#define PACKAGE "openpa"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://trac.mcs.anl.gov/projects/openpa/newticket"

/* Define to the full name of this package. */
#define PACKAGE_NAME "OpenPA"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "OpenPA 1.0.3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "openpa"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.0.3"

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* define to 1 to force using lock-based atomic primitives */
/* #undef USE_LOCK_BASED_PRIMITIVES */

/* define to 1 if unsafe (non-atomic) primitives should be used */
/* #undef USE_UNSAFE_PRIMITIVES */

/* Version number of package */
#define VERSION "1.0.3"

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


