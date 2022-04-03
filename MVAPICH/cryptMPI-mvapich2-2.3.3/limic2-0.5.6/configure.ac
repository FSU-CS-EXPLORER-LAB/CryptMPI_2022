#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([LiMIC2], [0.5.6], [jinh@konkuk.ac.kr])
AC_CONFIG_SRCDIR([src/limic.h])
AM_INIT_AUTOMAKE([foreign])
 
AC_ARG_ENABLE([module],
              [AS_HELP_STRING([--enable-module],
                              [Build Limic2 kernel module])],
              [],
              [enable_module=no])

AC_ARG_WITH([kernel],
            [AS_HELP_STRING([--with-kernel=RELEASE],
                            [Specify kernel release to use for building LiMIC2
                             kernel module.  Defaults to currently running
                             kernel.])],
            [],
            [with_kernel=`uname -r`])

AC_ARG_WITH([kernel-mod],
            [AS_HELP_STRING([--with-kernel-mod=DIR],
                            [Path to install LiMIC2 kernel module.  Defaults to
                             external module location for the kernel the module
                             is built against.])],
            [],
            [with_kernel_mod=/lib/modules/$with_kernel/extra])

AC_SUBST([KERNEL_SRC], [/lib/modules/$with_kernel/build])
AC_SUBST([KERNEL_MOD], [$with_kernel_mod])

AM_CONDITIONAL([BUILD_KMOD], [test "x$enable_module" != "xno"])

# Checks for programs.
AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S
AC_PROG_LIBTOOL

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h inttypes.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T

# Checks for library functions.

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 module/limic])
AC_OUTPUT
