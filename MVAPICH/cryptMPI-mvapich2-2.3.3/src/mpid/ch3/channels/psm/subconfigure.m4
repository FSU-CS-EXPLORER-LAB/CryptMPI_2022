[#] start of __file__
dnl
dnl _PREREQ handles the former role of mpich2prereq, setup_device, etc
AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
AM_CONDITIONAL([BUILD_CH3_PSM],[test "X$device_name" = "Xch3" -a "X$channel_name" = "Xpsm"])
AM_COND_IF([BUILD_CH3_PSM],[
build_osu_mvapich=yes
AC_MSG_NOTICE([RUNNING PREREQ FOR ch3:psm])
MPID_MAX_THREAD_LEVEL=MPI_THREAD_MULTIPLE
])dnl end AM_COND_IF(BUILD_CH3_PSM,...)
])dnl
dnl
dnl _BODY handles the former role of configure in the subsystem
AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[
AM_COND_IF([BUILD_CH3_PSM],[
AC_MSG_NOTICE([RUNNING CONFIGURE FOR ch3:psm])

AC_CHECK_HEADERS(           \
    netdb.h                 \
    sys/ioctl.h             \
    sys/socket.h            \
    sys/sockio.h            \
    sys/types.h             \
    errno.h)

AC_ARG_WITH(psm, [--with-psm=path - specify path where psm include directory and lib directory can be found],
        if test "${with_psm}" != "yes" -a "${with_psm}" != "no" ; then
            LDFLAGS="$LDFLAGS -L${with_psm}/lib64 -L${with_psm}/lib"
                CPPFLAGS="$CPPFLAGS -I${with_psm}/include"
                fi,)
AC_ARG_WITH(psm-include, [--with-psm-include=path - specify path to psm include directory],
        if test "${with_psm_include}" != "yes" -a "${with_psm_include}" != "no" ; then
            CPPFLAGS="$CPPFLAGS -I${with_psm_include}"
            fi,)
AC_ARG_WITH(psm-lib, [--with-psm-lib=path - specify path to psm lib directory],
        if test "${with_psm_lib}" != "yes" -a "${with_psm_lib}" != "no" ; then
            LDFLAGS="$LDFLAGS -L${with_psm_lib}"
            fi,)

AC_ARG_WITH(psm2, [--with-psm2=path - specify path where psm2 include directory and lib directory can be found],
        if test "${with_psm2}" != "yes" -a "${with_psm2}" != "no" ; then
            LDFLAGS="$LDFLAGS -L${with_psm2}/lib64 -L${with_psm2}/lib"
            CPPFLAGS="$CPPFLAGS -I${with_psm2}/include"
            fi,)
AC_ARG_WITH(psm2-include, [--with-psm2-include=path - specify path to psm2 include directory],
        if test "${with_psm2_include}" != "yes" -a "${with_psm2_include}" != "no" ; then
            CPPFLAGS="$CPPFLAGS -I${with_psm2_include}"
            fi,)
AC_ARG_WITH(psm2-lib, [--with-psm2-lib=path - specify path to psm2 lib directory],
        if test "${with_psm2_lib}" != "yes" -a "${with_psm2_lib}" != "no" ; then
            LDFLAGS="$LDFLAGS -L${with_psm2_lib}"
            fi,)

if test "x$with_psm" != "x" -a "x$with_psm2" != "x" ; then
    AC_MSG_ERROR([Cannot specify option for --with-psm and --with-psm2 simultaneously])
elif test "x$with_psm2" != "x" ; then
    AC_CHECK_HEADER([psm2.h], , [
     AC_MSG_ERROR(['psm2.h not found.  Did you specify --with-psm2= or --with-psm2-include=?'])
    ])
    AC_CHECK_LIB(psm2, psm2_init, , [
    AC_MSG_ERROR(['psm2 library not found.  Did you specify --with-psm2= or --with-psm2-lib=?'])
    ])
elif test "x$with_psm" != "x" ; then
    AC_CHECK_HEADER([psm.h], , [
     AC_MSG_ERROR(['psm.h not found.  Did you specify --with-psm= or --with-psm-include=?'])
    ])
    AC_CHECK_LIB(psm_infinipath, psm_init, , [
     AC_MSG_ERROR(['psm_infinipath library not found.  Did you specify --with-psm= or --with-psm-lib=?'])
    ])
else
    AC_CHECK_HEADER([psm.h], [have_psm=yes], [have_psm=no])
    AC_CHECK_HEADER([psm2.h], [have_psm2=yes], [have_psm2=no])

    if test "$have_psm" == "no" -a "$have_psm2" == "no" ; then
        AC_MSG_ERROR(['No PSM header files found in default locations. Please retry after setting --with-psm= OR --with-psm2= to correct paths'])
    elif test "$have_psm2" == "yes" ; then
        AC_CHECK_LIB(psm2, psm2_init, , [
            AC_MSG_ERROR(['psm2 library not found.  Did you specify --with-psm2= or --with-psm2-lib=?'])
        ])
    else
        AC_CHECK_LIB(psm_infinipath, psm_init, , [
            AC_MSG_ERROR(['psm_infinipath library not found.  Did you specify --with-psm= or --with-psm-lib=?'])
        ])
    fi
fi


AC_CHECK_HEADER([infiniband/verbs.h],, [
    AC_MSG_WARN(['infiniband/verbs.h not found. Did you specify --with-ib-include=? Proceeding without libibverbs support.'])
])
AC_CHECK_LIB([ibverbs], [ibv_open_device],, [
    AC_MSG_WARN(['libibverbs not found. Did you specify --with-ib-libpath=? Proceeding without libibverbs support.'])
])

#which shared memory primitives to use
AC_ARG_WITH(shared-memory, [--with-shared-memory[=auto|sysv|mmap] - create shared memory using sysv or mmap (default is auto)],,
    with_shared_memory=auto)

if test "$with_shared_memory" = auto -o "$with_shared_memory" = mmap; then
    found_mmap_funcs=yes
    AC_CHECK_FUNCS(mmap munmap, , found_mmap_funcs=no)
    if test "$found_mmap_funcs" = yes ; then
        with_shared_memory=mmap
        AC_DEFINE(USE_MMAP_SHM,1,[Define if we have sysv shared memory])
        AC_MSG_NOTICE([Using a memory-mapped file for shared memory])
    elif test "$with_shared_memory" = mmap ; then
        AC_MSG_ERROR([cannot support shared memory:  mmap() or munmap() not found])
    fi
fi
if test "$with_shared_memory" = auto -o "$with_shared_memory" = sysv; then
    found_sysv_shm_funcs=yes
    AC_CHECK_FUNCS(shmget shmat shmctl shmdt, , found_sysv_shm_funcs=no)
    if test "$found_sysv_shm_funcs" = yes ; then
        AC_DEFINE(USE_SYSV_SHM,1,[Define if we have sysv shared memory])
        AC_MSG_NOTICE([Using SYSV shared memory])
    elif test "$with_shared_memory" = sysv ; then
        AC_MSG_ERROR([cannot support shared memory:  sysv shared memory functions functions not found])
    else
        AC_MSG_ERROR([cannot support shared memory:  need either sysv shared memory functions or mmap in order to support shared memory])
    fi
fi

if test "$found_sysv_shm_funcs" = yes ; then
   AC_CHECK_FUNCS(strtoll, , AC_MSG_ERROR([cannot find strtoll function needed by sysv shared memory implementation]))
fi

AC_DEFINE([_OSU_MVAPICH_], [1], [Define to enable MVAPICH2 customizations])
AC_DEFINE([CHANNEL_PSM], [1], [Define if using the psm channel])

## below is code that formerly lived in configure.ac
])dnl end AM_COND_IF(BUILD_CH3_PSM,...)
])dnl end _BODY
[#] end of __file__
