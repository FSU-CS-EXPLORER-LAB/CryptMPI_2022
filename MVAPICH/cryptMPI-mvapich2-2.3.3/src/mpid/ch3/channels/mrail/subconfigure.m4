[#] start of __file__
dnl
dnl _PREREQ handles the former role of mpich2prereq, setup_device, etc
AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX, [
    AS_IF([test "X$device_name" = "Xch3" -a "X$channel_name" = "Xmrail"],
          [build_mrail=yes
           build_osu_mvapich=yes],
          [build_mrail=no])
    AM_CONDITIONAL([BUILD_MRAIL], [test $build_mrail = yes])
    AM_COND_IF([BUILD_MRAIL], [
        AC_MSG_NOTICE([RUNNING PREREQ FOR ch3:mrail])
        MPID_MAX_THREAD_LEVEL=MPI_THREAD_MULTIPLE
    ])dnl end AM_COND_IF(BUILD_MRAIL,...)
])dnl
dnl
dnl _BODY handles the former role of configure in the subsystem
AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[
AM_COND_IF([BUILD_MRAIL], [
dnl
dnl user options
dnl
AC_ARG_WITH([rdma],
    [AS_HELP_STRING([--with-rdma], [specify the RDMA type])],
    [],
    [with_rdma=gen2])

AC_ARG_WITH([ib-include],
    [AS_HELP_STRING([--with-ib-include=@<:@Infiniband include path@:>@],
                       [Specify path to Infiniband header files])
    ],
    [AS_CASE([$with_ib_include],
        [yes|no], [AC_MSG_FAILURE([--with-ib-include must be passed a path])],
        [CPPFLAGS="$CPPFLAGS -I$with_ib_include"])
    ])

AC_ARG_WITH([ib-libpath],
    [AS_HELP_STRING([--with-ib-libpath=@<:@Infiniband library path@:>@],
                       [Specify path to Infiniband library files])
    ],
    [AS_CASE([$with_ib_libpath],
        [yes|no], [AC_MSG_FAILURE([--with-ib-libpath must be passed a path])],
        [LDFLAGS="$LDFLAGS -L$with_ib_libpath"])
    ])

dnl
dnl check build environment
dnl
AC_CHECK_HEADERS(                       \
    errno.h                             \
    stdlib.h                            \
    unistd.h                            \
    pthread.h                           \
    openacc.h                           \
)

AC_SEARCH_LIBS([pthread_create], [pthread])
AS_IF([test $ac_cv_search_pthread_create = no], [
    AC_MSG_ERROR([libpthread not found])
    ])

AC_CHECK_HEADERS([sys/syscall.h syscall.h], [
                  AC_CHECK_FUNCS([syscall])
                  break
                  ])

AC_ARG_WITH([scr],
    [AS_HELP_STRING([--with-scr],
        [Enable use of Scalable Checkpoint / Restart (SCR) for checkpointing])
    ],
    [AS_IF([test x$with_scr = xyes], [
        AC_DEFINE([ENABLE_SCR], [1], [Define to use SCR for checkpointing])
        AC_SEARCH_LIBS([crc32], [z])
        AS_IF([test $ac_cv_search_crc32 = no], [
            AC_MSG_ERROR([libz not found (required by scr)])
            ])
        ])
    ],
    [with_scr=no])

AC_ARG_VAR([PDSH_EXE], [path to pdsh command])
AC_PATH_PROG([PDSH_EXE], [pdsh], [notfound])

AC_ARG_VAR([DSHBAK_EXE], [path to dshbak command])
AC_PATH_PROG([DSHBAK_EXE], [dshbak], [notfound])

AS_IF([test x$with_scr != xno],
        [AS_IF([test x$PDSH_EXE = xnotfound],
            [AC_MSG_ERROR([pdsh not found, please set PDSH_EXE appropriately or disable SCR support])
            ])
         AS_IF([test x$DSHBAK_EXE = xnotfound],
            [AC_MSG_ERROR([dshbak not found, please set DSHBAK_EXE appropriately or disable SCR support])
            ])
        ])

AC_ARG_WITH([cma],
    [AS_HELP_STRING([--without-cma],
        [Disable use of CMA for intra-node communication])
    ],
    [],
    [with_cma=yes])

AC_ARG_WITH([limic2],
    [AS_HELP_STRING([--with-limic2=@<:@LiMIC2 installation path@:>@],
                    [Enable use of LiMIC2 for intra-node communication])
    ],
    [case $with_limic2 in
         yes|no) ;;
         *) with_limic2_include="$with_limic2/include" ;
            with_limic2_libpath="$with_limic2/lib" ;;
     esac
    ],
    [with_limic2=no])

AC_ARG_WITH([limic2-include],
    [AS_HELP_STRING([--with-limic2-include=@<:@LiMIC2 include path@:>@],
                       [Specify path to LiMIC2 header files])
    ], [
        case $with_limic2_include in
            yes|no) AC_MSG_FAILURE([must specify path!]);;
        esac
        test "x$with_limic2" != "xno" || with_limic2="yes"
    ])

AC_ARG_WITH([limic2-libpath],
    [AS_HELP_STRING([--with-limic2-libpath=@<:@LiMIC2 library path@:>@],
                       [Specify path to LiMIC2 library])
    ], [
        case $with_limic2_libpath in
            yes|no) AC_MSG_FAILURE([must specify path!]);;
        esac
        test "x$with_limic2" != "xno" || with_limic2="yes"
    ])

AS_IF([test x$with_cma != xno],
        [AC_DEFINE([_SMP_CMA_], [1],
            [Define to enable intra-node communication via CMA])
         AC_CHECK_FUNCS([process_vm_readv])
        ])

if test "x$with_limic2" != "xno"; then
    limic2_path_set="$with_limic2_include$with_limic2_libpath"

    test -n "$with_limic2_include" &&\
    CPPFLAGS="$CPPFLAGS -I$with_limic2_include"
    test -n "$with_limic2_libpath" &&\
    LDFLAGS="$LDFLAGS -L$with_limic2_libpath"

    AC_CHECK_HEADER([limic.h], [
                         limic_h_found=yes
                         AC_DEFINE([_SMP_LIMIC_], 1, [Define when using LiMIC2])
                     ])

    if test -n "$limic2_path_set" -a "x$limic_h_found" != "xyes"; then
        AC_MSG_ERROR([could not find limic.h with given include path!])
    fi

    AC_CHECK_LIB([limic2], [limic_open], [
            LIBS="-llimic2 $LIBS"
            limic2_lib_found=yes
        ])

    if test -n "$limic2_path_set" -a "x$limic2_lib_found" != "xyes"; then
        AC_MSG_ERROR([could not find LiMIC2 library with given libpath!])
    fi

    if test "x$limic_h_found$limic2_lib_found" != "xyesyes"; then
        AC_MSG_ERROR([cannot find LiMIC2, please install and retry.])
    fi

    AC_RUN_IFELSE([AC_LANG_PROGRAM([], [])],
        [],
        [AC_MSG_ERROR(Unable to run program linked with LiMIC2)],
        [AC_MSG_WARN([Unable to verify if program can run linked with LiMIC2])])

fi

AC_ARG_ENABLE([mcast],
              [AS_HELP_STRING([--enable-mcast],
                              [enable hardware multicast])
              ],
              [],
              [enable_mcast=yes])

AC_ARG_ENABLE([multi-subnet],
              [AS_HELP_STRING([--enable-multi-subnet],
                              [enable hardware multicast])
              ],
              [],
              [enable_multi_subnet=no])

AC_ARG_ENABLE([sharp],
              [AS_HELP_STRING([--enable-sharp],
                              [enable SHARP support])
              ],
              [enable_sharp=yes],
              [enable_sharp=no])

AC_ARG_ENABLE([3dtorus-support],
              [AS_HELP_STRING([--enable-3dtorus-support],
                              [Enable support for 3D torus networks])
              ],
              [],
              [enable_3dtorus_support=no])

AC_ARG_ENABLE([ckpt],
              [AS_HELP_STRING([--enable-ckpt],
                              [enable checkpoint/restart])
              ],
              [],
              [enable_ckpt=default])

AC_ARG_ENABLE([ckpt-aggregation],
              [AS_HELP_STRING([--enable-ckpt-aggregation],
                              [enable aggregation with checkpoint/restart])
              ],
              [],
              [enable_ckpt_aggregation=check])

AC_ARG_ENABLE([ckpt-migration],
              [AS_HELP_STRING([--enable-ckpt-migration],
                              [enable process migration])
              ],
              [],
              [enable_ckpt_migration=no])

AC_ARG_WITH([ftb],
            [AS_HELP_STRING([--with-ftb@[:@=path@:]@],
                            [provide path to ftb package])
            ],
            [],
            [with_ftb=check])

AC_ARG_WITH([ftb-include],
            [AS_HELP_STRING([--with-ftb-include=@<:@path@:>@],
                            [specify the path to the ftb header files])
            ],
            [AS_CASE([$with_ftb_include],
                     [yes|no], [AC_MSG_ERROR([arg to --with-ftb-include must be a path])])
            ],
            [])

AC_ARG_WITH([ftb-libpath],
            [AS_HELP_STRING([--with-ftb-libpath=@<:@path@:>@],
                            [specify the path to the ftb library])
            ],
            [AS_CASE([$with_ftb_libpath],
                     [yes|no], [AC_MSG_ERROR([arg to --with-ftb-libpath must be a path])])
            ],
            [])

AC_ARG_WITH([blcr],
            [AS_HELP_STRING([--with-blcr@[:@=path@:]@],
                            [provide path to blcr package])
            ],
            [],
            [with_blcr=check])

AC_ARG_WITH([blcr-include],
            [AS_HELP_STRING([--with-blcr-include=@<:@path@:>@],
                            [specify the path to the blcr header files])
            ],
            [AS_CASE([$with_blcr_include],
                     [yes|no], [AC_MSG_ERROR([arg to --with-blcr-include must be a path])])
            ],
            [])

AC_ARG_WITH([blcr-libpath],
            [AS_HELP_STRING([--with-blcr-libpath=@<:@path@:>@],
                            [specify the path to the blcr library])
            ],
            [AS_CASE([$with_blcr_libpath],
                     [yes|no], [AC_MSG_ERROR([arg to --with-blcr-libpath must be a path])])
            ],
            [])

AC_ARG_WITH(cluster-size,
[--with-cluster-size=level - Specify the cluster size.],,with_cluster_size=small)
AC_ARG_WITH(dapl-include,
[--with-dapl-include=path - Specify the path to the DAPL header files.],,)
AC_ARG_WITH(dapl-libpath,
[--with-dapl-libpath=path - Specify the path to the dapl library.],,with_dapl_libpath=default)
AC_ARG_WITH(dapl-provider,
[--with-dapl-provider=type - Specify the dapl provider.],,with_dapl_provider=default)
AC_ARG_WITH(dapl-version,
[--with-dapl-version=version - Specify the dapl version.],,with_dapl_version=2.0)
AC_ARG_ENABLE(header-caching,
[--enable-header-caching - Enable header caching.],,enable_header_caching=yes)
AC_ARG_WITH(ib-include,
[--with-ib-include=path - Specify the path to the InfiniBand header files.],,)
AC_ARG_WITH(ib-libpath,
[--with-ib-libpath=path - Specify the path to the infiniband libraries.],,with_ib_libpath=default)
AC_ARG_WITH(io-bus,
[--with-io-bus=type - Specify the i/o bus type.],,with_io_bus=PCI_EX) 
AC_ARG_WITH(pmi,
[--with-pmi=name - Specify the PMI interface.],,)
AC_ARG_ENABLE(rdma-cm,
[--enable-rdma-cm - Enable support for RDMA CM.],,enable_rdma_cm=default)
AC_ARG_ENABLE(registration-cache,
[--enable-registration-cache - Enable registration caching on Linux.],,enable_registration_cache=default)

AC_ARG_ENABLE([xrc],
              [AS_HELP_STRING([--disable-xrc],
                              [compile MVAPICH2 without XRC support])
              ],
              [],
              [enable_xrc=check])

AC_ARG_WITH([libcuda],
        [AS_HELP_STRING([--with-libcuda=@<:@libcuda directory path@:>@],
            [Specify path of directory containing libcuda])
        ],
        [LDFLAGS="-L$with_libcuda $LDFLAGS"])

AC_ARG_WITH([libcudart],
        [AS_HELP_STRING([--with-libcudart=@<:@libcudart directory path@:>@],
            [Specify path of directory containing libcudart])
        ],
        [LDFLAGS="-L$with_libcudart $LDFLAGS"])

AC_ARG_ENABLE([cuda],
        [AS_HELP_STRING([--enable-cuda],
            [enable MVAPICH-GPU design (default is no); you may specify
            --enable-cuda=basic to enable the basic MVAPICH-GPU design without
            the optimized cuda kernel data movement routines])
        ],
        [],
        [enable_cuda=no])

AS_CASE([$enable_cuda],
        [yes], [build_mrail_cuda=yes; build_mrail_cuda_kernels=yes],
        [basic], [build_mrail_cuda=yes])

AS_IF([test x$build_mrail_cuda_kernels = xyes],
        [AC_DEFINE([USE_GPU_KERNEL], [1], [Define to enable cuda kernel functions])])

AS_IF([test "x$enable_3dtorus_support" != xno],
      [AC_DEFINE([ENABLE_3DTORUS_SUPPORT], [1],
                 [Define to enable 3D Torus support])])

AS_CASE([$with_ftb],
        [yes|no|check], [],
        [with_ftb_include="$with_ftb/include"
         with_ftb_libpath="$with_ftb/lib"
         with_ftb=yes])

AS_IF([test -n "$with_ftb_include"],
      [CPPFLAGS="$CPPFLAGS -I$with_ftb_include"
       with_ftb=yes])

AS_IF([test -n "$with_ftb_libpath"],
      [LDFLAGS="$LDFLAGS -L$with_ftb_libpath"
       mrail_ld_library_path="$with_ftb_libpath:$mrail_ld_library_path"
       FFLAGS="-L$with_ftb_libpath $FFLAGS"
       LDFLAGS="-L$with_ftb_libpath $LDFLAGS"
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
      [LDFLAGS="$LDFLAGS -L$with_blcr_libpath"
       mrail_ld_library_path="$with_blcr_libpath:$mrail_ld_library_path"
       FFLAGS="-L$with_blcr_libpath $FFLAGS"
       LDFLAGS="-L$with_blcr_libpath $LDFLAGS"
       with_blcr=yes])

AS_IF([test "x$enable_ckpt" = xdefault], [
       AS_IF([test "x$enable_ckpt_aggregation" = xyes || test "x$enable_ckpt_migration" = xyes], [enable_ckpt=yes], [enable_ckpt=no])
       ])


AS_IF([test "x$with_blcr" = "xno"], [
       AS_IF([test "x$enable_ckpt" = "xyes"], [AC_MSG_ERROR([BLCR is required if Checkpoint/Restart is enabled])])
       AS_IF([test "x$enable_ckpt_migration" = "xyes"], [AC_MSG_ERROR([BLCR is required if Checkpoint/Restart Migration is enabled])])
       ])

AS_IF([test "x$with_ftb" = "xno"], [
       AS_IF([test "x$enable_ckpt_migration" = "xyes"], [AC_MSG_ERROR([FTB is required if Checkpoint/Restart Migration is enabled])])
       ])
        
AS_IF([test "x$enable_ckpt_migration" = "xyes"], [
       AS_IF([test "x$with_ftb" = "xcheck"], [with_ftb=yes])
       ])

AS_IF([test "x$enable_ckpt" = xyes], [
       AC_MSG_NOTICE([Testing Checkpoint/Restart dependencies])
       AC_CHECK_HEADER([libcr.h],
                       [],
                       [AC_MSG_ERROR(['libcr.h not found. Please specify --with-blcr-include'])])
       AC_SEARCH_LIBS([cr_init],
                      [cr],
                      [],
                      [AC_MSG_ERROR([libcr not found.])],
                      [])

       AC_DEFINE(CKPT, 1, [Define to enable Checkpoint/Restart support.])

       AC_MSG_CHECKING([whether to enable support for FTB-CR])
       AC_MSG_RESULT($enable_ftb_cr)

       AS_IF([test "x$enable_ckpt_migration" = xyes], [
              AC_CHECK_HEADER([attr/xattr.h],
                              [],
                              [AC_MSG_ERROR(['attr/xattr.h not found.  Needed for migration support'])])
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


if test -n "`echo $build_os | grep linux`"; then
    if test "$build_cpu" = "i686"; then
        AC_DEFINE(_IA32_, 1, [Define to specify the build CPU type.])
    elif test "$build_cpu" = "ia64"; then
        AC_DEFINE(_IA64_, 1, [Define to specify the build CPU type.])
    elif test "$build_cpu" = "x86_64"; then
        if test "`grep 'model name' </proc/cpuinfo | grep Opteron`"; then
            AC_DEFINE(_X86_64_, 1, [Define to specify the build CPU type.])

            if test "`grep 'siblings' </proc/cpuinfo | head -1 | awk '{ print $3 '}`" = "4"; then
                AC_DEFINE(_AMD_QUAD_CORE_, 1, [Define to specify the build CPU is an AMD quad core.])
            fi
        else
            AC_DEFINE(_EM64T_, 1, [Define to specify the build CPU type.])
        fi
    else
        AC_MSG_WARN([The build CPU type may not be supported.])
    fi

    if test "$enable_registration_cache" = "default"; then
        enable_registration_cache=yes
    fi
elif test -n "`echo $build_os | grep solaris`"; then
    if test "$build_cpu" != "i386"; then
        AC_MSG_ERROR([The build CPU type is not supported.])
    fi

    AC_DEFINE(SOLARIS, 1, [Define to specify the build OS type.])

    if test "$enable_registration_cache" != "default"; then
        AC_MSG_ERROR([Registration caching is not configurable on Solaris.])
    fi

    AC_DEFINE(DISABLE_PTMALLOC,1,[Define to disable use of ptmalloc. On Linux, disabling ptmalloc also disables registration caching.])
else
    AC_MSG_ERROR([The build OS type is not supported.])
fi

AS_IF([test "x$enable_ckpt" = xyes], [
       AS_CASE([$enable_rdma_cm],
               [default], [enable_rdma_cm=no],
               [yes], [AC_MSG_ERROR([RDMA CM is not supported with BLCR.])])
       AS_IF([test "x$enable_registration_cache" = "xno"], [
              AC_MSG_ERROR([Registration caching is required for BLCR support.])
             ])
      ])

if test "$with_rdma" = "gen2"; then
    AC_MSG_CHECKING([for the InfiniBand includes path])
    if test -n "$with_ib_include"; then
        AC_MSG_RESULT($with_ib_include)
	CPPFLAGS="-I${with_ib_include} $CPPFLAGS"
    else
        AC_MSG_RESULT([default])
    fi

    AC_MSG_CHECKING([for the InfiniBand library path])

    if test "$with_ib_libpath" != "default"; then
        if test ! -d $with_ib_libpath; then
            AC_MSG_ERROR([The specified InfiniBand library path is invalid.])
        fi
    else
        ofed64=/usr/local/ofed/lib64
        ofed=/usr/local/ofed/lib

        if test -d $ofed64; then
            with_ib_libpath=$ofed64
        elif test -d $ofed; then
            with_ib_libpath=$ofed
        fi
    fi

    if test "$with_ib_libpath" != "default"; then
        mrail_ld_library_path=${with_ib_libpath}:$mrail_ld_library_path
        FFLAGS="-L${with_ib_libpath} $FFLAGS"
        LDFLAGS="-L${with_ib_libpath} $LDFLAGS"
    fi

    AC_MSG_RESULT($with_ib_libpath)

    AC_SEARCH_LIBS([shm_open], [rt])
    AS_CASE([$ac_cv_search_shm_open],
            ["none required"],,
            ["no"], [AC_MSG_ERROR(['no library containing `shm_open\' was found'])],
            [])

    AC_SEARCH_LIBS(dlopen, dl,,[AC_MSG_ERROR(['libdl not found'])])
    AC_SEARCH_LIBS(ibv_open_device, ibverbs,[AC_DEFINE([HAVE_LIBIBVERBS], [1])],[AC_MSG_ERROR(['libibverbs not found. Did you specify --with-ib-libpath=?'])],)
    AC_CHECK_HEADER([infiniband/verbs.h],,[AC_MSG_ERROR(['infiniband/verbs.h not found. Did you specify --with-ib-include=?'])])

    AC_CHECKING([checking for infiniband umad installation])
    AC_SEARCH_LIBS(umad_init, ibumad, lib_umad_found="yes", lib_umad_found="no",)
    if test $lib_umad_found = "yes"; then
        AC_CHECK_HEADER([infiniband/umad.h],,lib_umad_found="no")
        if test $lib_umad_found = "yes"; then
            AC_DEFINE(HAVE_LIBIBUMAD, 1, [UMAD installation found.])
            AC_MSG_NOTICE([infiniband libumand found])
        else 
            AC_MSG_NOTICE([infiniband libumand not found])
        fi
    else
        AC_MSG_NOTICE([infiniband libumand not found])
    fi

    # Check for available functions after finding header files
    AC_CHECK_FUNCS([ibv_open_xrc_domain])
   
    AS_CASE([$ac_cv_func_ibv_open_xrc_domain],
            [no], [AS_CASE([$enable_xrc],
                           [check], [AC_MSG_WARN([support for XRC is disabled])],
                           [yes], [AC_MSG_ERROR([infiniband/verbs.h does not provide support for XRC])])],
            [yes], [AS_CASE([$enable_xrc],
                            [no], [],
                            [AC_DEFINE([_ENABLE_XRC_], [1], [Define to enable XRC support])])])

    AC_MSG_CHECKING([whether to enable hybrid communication channel])
    if test "$enable_hybrid" == "yes"; then
        AC_DEFINE(_ENABLE_UD_,1,[Define to enable hybrid design.])
    fi
    AC_MSG_RESULT($enable_hybrid)

    if test "$enable_rdma_cm" = "yes"; then
        AC_CHECKING([for RDMA CM support])
        AC_SEARCH_LIBS(rdma_create_event_channel, rdmacm,,[AC_MSG_ERROR(['librdmacm not found. Did you specify --with-ib-libpath=?'])],)
        AC_CHECK_HEADER([rdma/rdma_cma.h],,[AC_MSG_ERROR(['rdma/rdma_cma.h not found. Did you specify --with-ib-include=?'])])
        AC_DEFINE(RDMA_CM, 1, [Define to enable support from RDMA CM.])
        AC_MSG_NOTICE([RDMA CM support enabled])
    elif test "$enable_rdma_cm" = "default"; then
        AC_CHECKING([for RDMA CM support])
        AC_SEARCH_LIBS(rdma_create_event_channel, rdmacm,enable_rdma_cm="yes",enable_rdma_cm="no",)
        if test $enable_rdma_cm = "yes"; then
            AC_CHECK_HEADER([rdma/rdma_cma.h],,enable_rdma_cm="no")
            if test $enable_rdma_cm = "yes"; then
                AC_DEFINE(RDMA_CM, 1, [Define to enable support from RDMA CM.])
                AC_MSG_NOTICE([RDMA CM support enabled])
            else
                AC_MSG_NOTICE([RDMA CM support disabled])
            fi
        else
            AC_MSG_NOTICE([RDMA CM support disabled])
        fi
    fi

    if test "$enable_multi_subnet" = "yes" ; then
        if test "$enable_rdma_cm" = "yes"; then
            AC_CHECKING([for inter subnet communication support])
            AC_CHECK_DECLS([RAI_FAMILY],[rai_family=yes],[rai_family=no],[[#include <rdma/rdma_cma.h>]])
            if test $rai_family = "yes"; then
                AC_DEFINE(_MULTI_SUBNET_SUPPORT_, 1 , [Define to enable inter subnet communication support.])
                AC_MSG_NOTICE([inter subnet communication support enabled])
            fi
        else
            AC_MSG_ERROR(['Support for inter subnet communication requires RDMA_CM support. Please retry with --enable-rdma-cm or after removing --enable-multi-subnet'])
        fi
    fi

    if test "$enable_mcast" != "no"; then
        AC_CHECKING([for Hardware multicast support enabled])
        AC_SEARCH_LIBS(mad_get_field, ibmad, lib_mad_found="yes", lib_mad_found="no",)
        AC_CHECK_HEADER([infiniband/mad.h],,[AC_MSG_ERROR(['infiniband/mad.h not found. Please retry with --disable-mcast'])])
        if test $lib_mad_found = "yes"; then
            AC_DEFINE(_MCST_SUPPORT_, 1 , [Define to enable Hardware multicast support.])
            AC_MSG_NOTICE([Hardware multicast support enabled])
        fi
    fi

    if test "$enable_sharp" != "no"; then
        AC_CHECKING([for SHARP support enabled])
        AC_CHECK_HEADER([api/sharp_coll.h],,[AC_MSG_ERROR(['api/sharp_coll.h  not found. Please retry without --enable-sharp'])])
        AC_DEFINE(_SHARP_SUPPORT_, 1 , [Define to enable switch IB-2 sharp support.])
        AC_MSG_NOTICE([SHARP support enabled])
    fi


    AC_ARG_WITH([cuda],
        [AS_HELP_STRING([--with-cuda=@<:@CUDA installation path@:>@],
                        [Specify path to CUDA installation])
        ],
        [case $with_cuda in
             yes|no) ;;
            *) CPPFLAGS="-I$with_cuda/include $CPPFLAGS" ;
                WRAPPER_CPPFLAGS="$WRAPPER_CPPFLAGS -I$with_cuda/include"
                LDFLAGS="-L$with_cuda/lib64 -L$with_cuda/lib $LDFLAGS" ;;
        esac
        ])

        AC_ARG_WITH([cuda-include],
        [AS_HELP_STRING([--with-cuda-include=@<:@CUDA include path@:>@],
                          [Specify path to CUDA header files])
        ], [
            case $with_cuda_include in
                yes|no) AC_MSG_FAILURE([must specify path!]) ;;
                *) CPPFLAGS="-I$with_cuda_include $CPPFLAGS"
                WRAPPER_CPPFLAGS="$WRAPPER_CPPFLAGS -I$with_cuda_include" ;;
            esac
        ])

    AC_ARG_WITH([cuda-libpath],
        [AS_HELP_STRING([--with-cuda-libpath=@<:@CUDA library path@:>@],
                           [Specify path to CUDA libraries])
        ], [
            case $with_cuda_libpath in
                yes|no) AC_MSG_FAILURE([must specify path!]) ;;
                *) LDFLAGS="-L$with_cuda_libpath $LDFLAGS" ;;
            esac
        ])

    AC_ARG_ENABLE([cuda],
                  [AS_HELP_STRING([--enable-cuda],
                                  [enable MVAPICH-GPU design])
                  ])

    AS_IF([test "$build_mrail_cuda" == "yes"], [
        AC_CHECK_HEADER([cuda.h],
                           [],
                           [AC_MSG_ERROR([Could not find cuda.h])])

        AC_SEARCH_LIBS([cuPointerGetAttribute], 
                          [cuda],
                          [],
                          [AC_MSG_ERROR([Could not link with cuda])])

        AC_SEARCH_LIBS([cudaFree],
                          [cudart],
                          [],
                          [AC_MSG_ERROR([Could not link with cudart])])

           AC_RUN_IFELSE([AC_LANG_PROGRAM([], [])],
                         [],
                         [AC_MSG_ERROR(Unable to run program linked with CUDA)],
                         [AC_MSG_WARN([Unable to verify if program can run linked with CUDA])])


           AC_DEFINE(_ENABLE_CUDA_,1,[Define to enable MVAPICH2-GPU design.])
           found_cudaipc_funcs=yes
        AC_CHECK_FUNCS([cudaIpcGetMemHandle], , found_cudaipc_funcs=no)
        if test "$found_cudaipc_funcs" = yes ; then
               AC_DEFINE(HAVE_CUDA_IPC,1,[Define to enable CUDA IPC features])
        fi
        ])

elif test "$with_rdma" = "udapl"; then
    :
else
    AC_MSG_ERROR([The specified RDMA type is not supported.])
fi

AC_MSG_CHECKING([whether to enable header caching])

if test "$enable_header_caching" != "yes"; then
    AC_DEFINE(MV2_DISABLE_HEADER_CACHING,1,[Define to disable header caching.])
fi

AC_MSG_RESULT($enable_header_caching)

if test -n "`echo $build_os | grep linux`"; then
    AC_MSG_CHECKING([whether to enable registration caching])

    if test "$enable_registration_cache" != "yes"; then
        AC_DEFINE(DISABLE_PTMALLOC,1,[Define to disable use of ptmalloc. On Linux, disabling ptmalloc also disables registration caching.])
    fi

    AC_MSG_RESULT($enable_registration_cache)
fi

AC_CHECK_FUNCS(snprintf)

if test "$ac_cv_func_snprintf" = "yes" ; then
    PAC_FUNC_NEEDS_DECL([#include <stdio.h>],snprintf)
fi

# check how to allocate shared memory
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

dnl
dnl Definitions based on user selections
dnl
AC_DEFINE([_OSU_MVAPICH_], [1], [Define to enable MVAPICH2 customizations])
AC_DEFINE([CHANNEL_MRAIL], [1], [Define if using the mrail channel])
AC_DEFINE([MPIDI_CH3_CHANNEL_RNDV], [1],
        [Define to enable channel rendezvous (Required by MVAPICH2)])
AC_DEFINE([MPID_USE_SEQUENCE_NUMBERS], [1],
        [Define to enable use of sequence numbers (Required by MVAPICH2)])
AC_DEFINE([_OSU_COLLECTIVES_], [1],
        [Define to enable the use of MVAPICH2 implmentation of collectives])

AS_CASE([$with_rdma],
    [gen2], [
    AC_DEFINE([CHANNEL_MRAIL_GEN2], [1],
        [Define if using the gen2 subchannel])
    ],
    [udapl], [AC_MSG_ERROR([This subchannel has been deprecated, please see ]
        [our userguide to determine the supported channels.])
    ],
    [AC_MSG_ERROR([Invalid RDMA type detected])])

## below is code that formerly lived in configure.ac
])dnl end AM_COND_IF(BUILD_MRAIL,...)

    dnl automake conditionals should not appear in conditional blocks as this
    dnl can cause confusion in the makefiles
    AM_CONDITIONAL([BUILD_MRAIL_GEN2], [test X$with_rdma = Xgen2])
    AM_CONDITIONAL([BUILD_MRAIL_OPENACC],
            [test X$ac_cv_header_openacc_h = Xyes -a X$build_mrail_cuda = Xyes])
    AM_CONDITIONAL([BUILD_MRAIL_CUDA], [test X$build_mrail_cuda = Xyes])
    AM_CONDITIONAL([BUILD_MRAIL_CUDA_KERNELS],
            [test X$build_mrail_cuda_kernels = Xyes])
    AM_CONDITIONAL([BUILD_USE_PGI], [`$CXX -V 2>&1 | grep pgc++ > /dev/null 2>&1`])
    AM_CONDITIONAL([BUILD_LIB_CH3AFFINITY], [test X$build_mrail = Xyes])
    AM_CONDITIONAL([BUILD_LIB_SCR], [test X$with_scr = Xyes])
    AM_CONDITIONAL([BUILD_LIB_CR], [test X$enable_ckpt = Xyes])
])dnl end _BODY
[#] end of __file__
