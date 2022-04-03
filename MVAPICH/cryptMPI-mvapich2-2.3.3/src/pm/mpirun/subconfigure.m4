[#] start of __file__

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
])

AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[

# the pm_names variable is set by the top level configure
build_pm_mpirun=no
for pm_name in $pm_names ; do
    if test "X$pm_name" = "Xmpirun" ; then
        build_pm_mpirun=yes
    fi
done

AM_CONDITIONAL([BUILD_PM_MPIRUN],[test "x$build_pm_mpirun" = "xyes"])

AM_COND_IF([BUILD_PM_MPIRUN],[

AC_ARG_ENABLE([rsh],
              [AS_HELP_STRING([--enable-rsh],
                              [Enable use of rsh for command execution by default.])
              ],
              [],
              [enable_rsh=no])

AC_ARG_VAR([RSH_CMD], [path to rsh command])
AC_PATH_PROG([RSH_CMD], [rsh], [/usr/bin/rsh])

AC_ARG_VAR([SSH_CMD], [path to ssh command])
AC_PATH_PROG([SSH_CMD], [ssh], [/usr/bin/ssh])

AC_ARG_VAR([ENV_CMD], [path to env command])
AC_PATH_PROG([ENV_CMD], [env], [/usr/bin/env])

AC_ARG_VAR([DBG_CMD], [path to debugger command])
AC_PATH_PROG([DBG_CMD], [gdb], [/usr/bin/gdb])

AC_ARG_VAR([XTERM_CMD], [path to xterm command])
AC_PATH_PROG([XTERM_CMD], [xterm], [/usr/bin/xterm])

AC_ARG_VAR([SHELL_CMD], [path to shell command])
AC_PATH_PROG([SHELL_CMD], [bash], [/bin/bash])

AC_ARG_VAR([TOTALVIEW_CMD], [path to totalview command])
AC_PATH_PROG([TOTALVIEW_CMD], [totalview], [/usr/totalview/bin/totalview])

AC_ARG_WITH([fuse],
    [AS_HELP_STRING([--with-fuse@[:@=path@:]@],
        [provide path to fuse package])
    ],
    [],
    [with_fuse=check])

AC_ARG_WITH([fuse-include],
    [AS_HELP_STRING([--with-fuse-include=@<:@path@:>@],
        [specify the path to the fuse header files])
    ],
    [AS_CASE([$with_fuse_include],
        [yes|no], [AC_MSG_ERROR([arg to --with-fuse-include must be a path])])
    ],
    [])

AC_ARG_WITH([fuse-libpath],
    [AS_HELP_STRING([--with-fuse-libpath=@<:@path@:>@],
        [specify the path to the fuse library])
    ],
    [AS_CASE([$with_fuse_libpath],
        [yes|no], [AC_MSG_ERROR([arg to --with-fuse-libpath must be a path])])
    ],
    [])

AC_PROG_YACC
AC_PROG_LEX

AC_SEARCH_LIBS(ceil, m,,[AC_MSG_ERROR([libm not found.])],)
AC_CHECK_FUNCS([strdup strndup get_current_dir_name])

if test -n "`echo $build_os | grep solaris`"; then
    AC_SEARCH_LIBS(herror, resolv,,[AC_MSG_ERROR([libresolv not found.])],)
    AC_SEARCH_LIBS(bind, socket,,[AC_MSG_ERROR([libsocket not found.])],)
    AC_SEARCH_LIBS(sendfile, sendfile,,[AC_MSG_ERROR([libsendfile not found.])],)
    mpirun_rsh_other_libs="-lresolv -lsocket"
    mpispawn_other_libs="-lresolv -lsocket -lnsl -lsendfile"
fi

AS_CASE([$with_ftb],
        [yes|no|check], [],
        [with_ftb_include="$with_ftb/include"
         with_ftb_libpath="$with_ftb/lib"
         with_ftb=yes])

AS_IF([test -n "$with_ftb_include"],
      [CPPFLAGS="$CPPFLAGS -I$with_ftb_include"
       with_ftb=yes])

AS_IF([test -n "$with_ftb_libpath"],
      [LDFLAGS="$LDFLAGS -L$with_ftb_libpath -Wl,-rpath,$with_ftb_libpath"
       with_ftb=yes])

AS_CASE([$with_blcr],
        [yes|no|check], [],
        [with_blcr_include="$with_blcr/include"
         with_blcr_libpath="$with_blcr/lib"
         with_blcr=yes])

AS_IF([test -n "$with_blcr_include"],
      [CPPFLAGS="$CPPFLAGS -I$with_blcr_include"
       with_blcr=yes])

AS_IF([test -n "$with_blcr_libpath"],
      [LDFLAGS="$LDFLAGS -L$with_blcr_libpath -Wl,-rpath,$with_blcr_libpath"
       with_blcr=yes])

AS_CASE([$with_fuse],
        [yes|no|check], [],
        [with_fuse_include="$with_fuse/include"
         with_fuse_libpath="$with_fuse/lib"
         with_fuse=yes])

AS_IF([test -n "$with_fuse_include"],
      [CPPFLAGS="$CPPFLAGS -I$with_fuse_include"
       with_fuse=yes])

AS_IF([test -n "$with_fuse_libpath"],
      [LDFLAGS="$LDFLAGS -L$with_fuse_libpath -Wl,-rpath,$with_fuse_libpath"
       with_fuse=yes])

AS_IF([test "x$enable_ckpt" = xdefault], [
       AS_IF([test "x$enable_ckpt_aggregation" = xyes || test "x$enable_ckpt_migration" = xyes], [enable_ckpt=yes], [enable_ckpt=no])
       ])

AS_IF([test "x$with_blcr" = "xno"], [
       AS_IF([test "x$enable_ckpt" = "xyes"], [AC_MSG_ERROR([BLCR is required if Checkpoint/Restart is enabled])])
       AS_IF([test "x$enable_ckpt_aggregation" = "xyes"], [AC_MSG_ERROR([BLCR is required if Checkpoint/Restart Aggregation is enabled])])
       AS_IF([test "x$enable_ckpt_migration" = "xyes"], [AC_MSG_ERROR([BLCR is required if Checkpoint/Restart Migration is enabled])])
       ])

AS_IF([test "x$with_fuse" = "xno"], [
       AS_IF([test "x$enable_ckpt_aggregation" = "xyes"], [AC_MSG_ERROR([FUSE is required if Checkpoint/Restart Aggregation is enabled])])
       ])

AS_IF([test "x$with_ftb" = "xno"], [
       AS_IF([test "x$enable_ckpt_migration" = "xyes"], [AC_MSG_ERROR([FTB is required if Checkpoint/Restart Migration is enabled])])
       ])
 
AS_IF([test "x$enable_ckpt_aggregation" = "xyes"], [
       AS_IF([test "x$with_fuse" = "xcheck"], [with_fuse=yes])
       ])

AS_IF([test "x$enable_ckpt_migration" = "xyes"], [
       AS_IF([test "x$with_ftb" = "xcheck"], [with_ftb=yes])
       ])

AC_MSG_CHECKING([whether to enable Checkpoint/Restart support support])
AC_MSG_RESULT([$enable_ckpt])

AS_IF([test "x$enable_ckpt" = xyes], [
       AC_CHECK_HEADER([libcr.h],
                       [],
                       [AC_MSG_ERROR(['libcr.h not found. Please specify --with-blcr-include'])])
       AC_SEARCH_LIBS([cr_init],
                      [cr],
                      [],
                      [AC_MSG_ERROR([libcr not found.])],
                      [])

       AC_DEFINE(CKPT, 1, [Define to enable Checkpoint/Restart support.])
       AS_IF([test "x$with_fuse" = xcheck || test "x$with_fuse" = xyes], [
              AC_MSG_NOTICE([checking checkpoint aggregation components])
              SAVE_LIBS="$LIBS"
              ckpt_aggregation=yes
              AC_SEARCH_LIBS([fuse_new],
                             [fuse],
                             [],
                             [AS_IF([test "x$with_fuse" = xyes], [AC_MSG_ERROR([fuse library not found])], [AC_MSG_WARN([fuse library not found])])
                              ckpt_aggregation=no])
              AC_SEARCH_LIBS([dlopen],
                             [dl],
                             [],
                             [AS_IF([test "x$with_fuse" = xyes], [AC_MSG_ERROR([dl library not found])], [AC_MSG_WARN([dl library not found])])
                              ckpt_aggregation=no])
              AC_SEARCH_LIBS([pthread_create],
                             [pthread],
                             [],
                             [AS_IF([test "x$with_fuse" = xyes], [AC_MSG_ERROR([pthread library not found])], [AC_MSG_WARN([pthread library not found])])
                              ckpt_aggregation=no])
              AC_SEARCH_LIBS([aio_read],
                             [rt],
                             [],
                             [AS_IF([test "x$with_fuse" = xyes], [AC_MSG_ERROR([rt library not found])], [AC_MSG_WARN([rt library not found])])
                              ckpt_aggregation=no])
              AS_IF([test "$ckpt_aggregation" = no], [
                     LIBS="$SAVE_LIBS"
                    ], [
                     with_fuse=yes
                    ])
             ])

       AS_IF([test "x$with_fuse" = xyes], [
              AC_DEFINE([CR_AGGRE], [1], [Define when using checkpoint aggregation])
              AC_DEFINE([_FILE_OFFSET_BITS], [64], [Define to set the number of file offset bits])
              AC_MSG_NOTICE([checkpoint aggregation enabled])
             ], [
              AC_MSG_WARN([checkpoint aggregation disabled])
             ])

       AC_MSG_CHECKING([whether to enable support for FTB-CR])
       AC_MSG_RESULT($enable_ftb_cr)

       AS_IF([test "x$enable_ckpt_migration" = xyes], [
              AC_CHECK_HEADER([libftb.h],
                              [],
                              [AC_MSG_ERROR(['libftb.h not found. Please specify --with-ftb-include'])])

              AC_SEARCH_LIBS([FTB_Connect],
                             [ftb],
                             [],
                             [AC_MSG_ERROR([libftb not found.])],
                             [])
              AC_DEFINE(CR_FTB, 1, [Define to enable FTB-CR support.])
             ])
      ])

if test "$enable_rsh" = "yes"; then
    AC_DEFINE(USE_RSH, 1, [Define to enable use of rsh for command execution by default.])
    AC_DEFINE(HAVE_PMI_IBARRIER, 1, [Define if pmi client supports PMI_Ibarrier])
    AC_DEFINE(HAVE_PMI_WAIT, 1, [Define if pmi client supports PMI_Wait])
    AC_DEFINE(HAVE_PMI2_KVS_IFENCE, 1, [Define if pmi client supports PMI2_KVS_Ifence])
    AC_DEFINE(HAVE_PMI2_KVS_WAIT, 1, [Define if pmi client supports PMI2_KVS_Wait])
    AC_DEFINE(HAVE_PMI2_SHMEM_IALLGATHER, 1, [Define if pmi client supports PMI2_Iallgather])
    AC_DEFINE(HAVE_PMI2_SHMEM_IALLGATHER_WAIT, 1, [Define if pmi client supports PMI2_Iallgather_wait])
    AC_DEFINE(HAVE_PMI2_SHMEM_IALLGATHER, 1, [Define if pmi client supports PMI2_SHMEM_Iallgather])
    AC_DEFINE(HAVE_PMI2_SHMEM_IALLGATHER_WAIT, 1, [Define if pmi client supports PMI2_SHMEM_Iallgather_wait])
fi

# MVAPICH2_VERSION is exported from the top level configure
AC_DEFINE_UNQUOTED([MVAPICH2_VERSION], ["$MVAPICH2_VERSION"], [Set to current version of mvapich2 package])

])

AM_CONDITIONAL([WANT_RDYNAMIC], [test "x$GCC" = xyes])
AM_CONDITIONAL([WANT_CKPT_RUNTIME], [test "x$enable_ckpt" = xyes])

dnl AC_MSG_NOTICE([RUNNING CONFIGURE FOR MPIRUN PROCESS MANAGERS])
# do nothing extra here for now

])dnl end _BODY

[#] end of __file__
