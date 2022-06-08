#ifndef _INCLUDE_MPLCONFIG_H
#define _INCLUDE_MPLCONFIG_H 1
 
/* include/mplconfig.h. Generated automatically at end of configure. */
/* include/config.h.  Generated from config.h.in by configure.  */
/* include/config.h.in.  Generated from configure.ac by autoheader.  */

/* define if valgrind is old and/or broken compared to what we are expecting
   */
#ifndef MPL_HAVE_BROKEN_VALGRIND 
#define MPL_HAVE_BROKEN_VALGRIND  1 
#endif

/* Define to 1 if you have the <ctype.h> header file. */
#ifndef MPL_HAVE_CTYPE_H 
#define MPL_HAVE_CTYPE_H  1 
#endif

/* Define to 1 if you have the <dlfcn.h> header file. */
#ifndef MPL_HAVE_DLFCN_H 
#define MPL_HAVE_DLFCN_H  1 
#endif

/* Define to 1 if you have the <drd.h> header file. */
/* #undef HAVE_DRD_H */

/* Define if GNU __attribute__ is supported */
#ifndef MPL_HAVE_GCC_ATTRIBUTE 
#define MPL_HAVE_GCC_ATTRIBUTE  1 
#endif

/* Define to 1 if you have the <helgrind.h> header file. */
/* #undef HAVE_HELGRIND_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef MPL_HAVE_INTTYPES_H 
#define MPL_HAVE_INTTYPES_H  1 
#endif

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

/* Define to 1 if you have the `putenv' function. */
#ifndef MPL_HAVE_PUTENV 
#define MPL_HAVE_PUTENV  1 
#endif

/* Define to 1 if you have the <search.h> header file. */
#ifndef MPL_HAVE_SEARCH_H 
#define MPL_HAVE_SEARCH_H  1 
#endif

/* Define to 1 if you have the `snprintf' function. */
#ifndef MPL_HAVE_SNPRINTF 
#define MPL_HAVE_SNPRINTF  1 
#endif

/* Define to 1 if you have the <stdarg.h> header file. */
#ifndef MPL_HAVE_STDARG_H 
#define MPL_HAVE_STDARG_H  1 
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

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef MPL_HAVE_UNISTD_H 
#define MPL_HAVE_UNISTD_H  1 
#endif

/* Define to 1 if you have the `usleep' function. */
/* #undef HAVE_USLEEP */

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

/* defined if the C compiler supports __typeof(variable) */
#ifndef MPL_HAVE___TYPEOF 
#define MPL_HAVE___TYPEOF  1 
#endif

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#ifndef MPL_LT_OBJDIR 
#define MPL_LT_OBJDIR  ".libs/" 
#endif

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

/* Define to 1 if you have the ANSI C header files. */
#ifndef MPL_STDC_HEADERS 
#define MPL_STDC_HEADERS  1 
#endif

/* Define if performing coverage tests */
/* #undef USE_COVERAGE */

/* Version number of package */
#ifndef MPL_VERSION 
#define MPL_VERSION  "0.1" 
#endif

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
