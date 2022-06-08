[#] start of __file__
dnl MPICH_SUBCFG_AFTER=src/mpid/ch3/channels/nemesis

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
    AM_COND_IF([BUILD_CH3_NEMESIS],[
        for net in $nemesis_networks ; do
            AS_CASE([$net],[ib],[build_nemesis_netmod_ib=yes
                                 build_osu_mvapich=yes])
        done
    ])
    AM_CONDITIONAL([BUILD_NEMESIS_NETMOD_IB],[test "X$build_nemesis_netmod_ib" = "Xyes"])
])dnl

AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[
# nothing to do for ib right now
AM_COND_IF([BUILD_NEMESIS_NETMOD_IB],[
AC_MSG_NOTICE([RUNNING CONFIGURE FOR ch3:nemesis:ib])

PAC_SET_HEADER_LIB_PATH(ibverbs)
AC_CHECK_HEADERS([sys/syscall.h syscall.h], [
                  AC_CHECK_FUNCS([syscall])
                  break
                  ])

AC_ARG_ENABLE(registration-cache,
[--enable-registration-cache - Enable registration caching on Linux.],,enable_registration_cache=default)

AS_IF([test -n "`echo $build_os | grep linux`"],
      [AS_IF([test "$enable_registration_cache" = "default"], [enable_registration_cache=yes])],
      [test -n "`echo $build_os | grep solaris`"],
      [AS_IF([test "$enable_registration_cache" != "default"], [AC_MSG_ERROR([Registration caching is not configurable on Solaris.])])])

AC_MSG_CHECKING([whether to enable registration caching])
AS_IF([test "$enable_registration_cache" != "yes"],
      [AC_DEFINE(DISABLE_PTMALLOC,1,[Define to disable use of ptmalloc. On Linux, disabling ptmalloc also disables registration caching.])
       enable_registration_cache=no
      ])
AC_MSG_RESULT($enable_registration_cache)

PAC_CHECK_HEADER_LIB_FATAL([ibverbs],
                           [infiniband/verbs.h],
                           [ibverbs],
                           [ibv_query_device])
have_umad=yes;

PAC_CHECK_HEADER_LIB(infiniband/umad.h, ibumad, umad_init, have_umad=yes, have_umad=no)
if test "$have_umad" = "yes" ; then
    AC_DEFINE(HAVE_LIBIBUMAD, 1, [UMAD installation found.])
else
    AC_MSG_NOTICE([infiniband libumand not found])
fi 

AC_SEARCH_LIBS([dlsym],
               [dl],
               [],
               [AC_MSG_ERROR([dlsym not available])])

AC_DEFINE([_OSU_MVAPICH_], [1], [Define to enable MVAPICH2 customizations])
AC_DEFINE([CHANNEL_NEMESIS_IB], [1], [Define if using the nemesis ib netmod])

])dnl end AM_COND_IF(BUILD_NEMESIS_NETMOD_IB,...)
])dnl end _BODY

[#] end of __file__
