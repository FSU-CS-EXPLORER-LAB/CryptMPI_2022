[#] start of __file__

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
])

AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[

# the pm_names variable is set by the top level configure
build_pm_flux=no
for pm_name in $pm_names ; do
    if test "X$pm_name" = "Xflux" ; then
        build_pm_flux=yes
    fi
done

AM_CONDITIONAL([BUILD_PM_FLUX],[test "x$build_pm_flux" = "xyes"])

AM_COND_IF([BUILD_PM_FLUX],[

# with_pmi is set by the top level configure
    AC_DEFINE([FLUX_PMI_CLIENT], [1], [Define if using flux pmi client])

AC_ARG_WITH([flux],
    [AS_HELP_STRING([--with-flux=@<:@Path to flux library@:>@],
                       [Specify path to PMI4PMIX installation])
    ],
    [AS_CASE([$with_flux],
        [yes|no], [AC_MSG_FAILURE([--with-flux must be passed a path])],
        [CPPFLAGS="$CPPFLAGS -I$with_flux/include/flux"]
        [LDFLAGS="$LDFLAGS -Wl,-rpath,$with_flux/lib/flux -L$with_flux/lib/flux"])
    ],
    [with_flux=no])

if test "x$with_pmi" = "xsimple" -o "x$with_pmi" = "xpmi1"; then
    AC_CHECK_HEADER([$with_flux/include/flux/pmi.h], [], [AC_MSG_ERROR([could not find flux/pmi.h. Please use --with-flux to point to the correct location. Configure aborted])])
    AC_CHECK_LIB([pmi], [PMI_Init],
             [PAC_PREPEND_FLAG([-lpmi],[LIBS])
              PAC_PREPEND_FLAG([-lpmi], [WRAPPER_LIBS])],
             [AC_MSG_ERROR([could not find the flux libpmi. Please use --with-flux to point to the correct location. Configure aborted])])
    AC_CHECK_FUNCS([PMI_Ibarrier], [AC_DEFINE([HAVE_PMI_IBARRIER], [1], [Define if pmi client supports PMI_Ibarrier])])
    AC_CHECK_FUNCS([PMI_Wait], [AC_DEFINE([HAVE_PMI_WAIT], [1], [Define if pmi client supports PMI_Wait])])
elif test "x$with_pmi" = "xpmi2/simple" -o "x$with_pmi" = "xpmi2"; then
    USE_PMI2_API=yes
    AC_CHECK_HEADER([$with_flux/include/flux/pmi2.h], [], [AC_MSG_ERROR([could not find flux/pmi2.h. Please use --with-flux to point to the correct location. Configure aborted])])
    AC_CHECK_LIB([pmi2], [PMI2_Init],
             [PAC_PREPEND_FLAG([-lpmi2],[LIBS])
              PAC_PREPEND_FLAG([-lpmi2], [WRAPPER_LIBS])],
             [AC_MSG_ERROR([could not find the flux libpmi2. Please use --with-flux to point to the correct location. Configure aborted])])
    AC_CHECK_FUNCS([PMI2_KVS_Ifence], [AC_DEFINE([HAVE_PMI2_KVS_IFENCE], [1], [Define if pmi client supports PMI2_KVS_Ifence])])
    AC_CHECK_FUNCS([PMI2_KVS_Wait], [AC_DEFINE([HAVE_PMI2_KVS_WAIT], [1], [Define if pmi client supports PMI2_KVS_Wait])])
    AC_CHECK_FUNCS([PMI2_Iallgather], [AC_DEFINE([HAVE_PMI2_SHMEM_IALLGATHER], [1], [Define if pmi client supports PMI2_Iallgather])])
    AC_CHECK_FUNCS([PMI2_Iallgather_wait], [AC_DEFINE([HAVE_PMI2_SHMEM_IALLGATHER_WAIT], [1], [Define if pmi client supports PMI2_Iallgather_wait])])
    AC_CHECK_FUNCS([PMI2_SHMEM_Iallgather], [AC_DEFINE([HAVE_PMI2_SHMEM_IALLGATHER], [1], [Define if pmi client supports PMI2_SHMEM_Iallgather])])
    AC_CHECK_FUNCS([PMI2_SHMEM_Iallgather_wait], [AC_DEFINE([HAVE_PMI2_SHMEM_IALLGATHER_WAIT], [1], [Define if pmi client supports PMI2_SHMEM_Iallgather_wait])])
elif test "x$with_pmi" = "xpmix"; then
    AC_ARG_WITH([pmix],
        [AS_HELP_STRING([--with-pmix=@<:@Path to pmix library@:>@],
                           [Specify path to PMIX installation])
        ],
        [AS_CASE([$with_pmix],
            [yes|no], [AC_MSG_FAILURE([--with-pmix must be passed a path])],
            [CPPFLAGS="$CPPFLAGS -I$with_pmix/include"]
            [LDFLAGS="$LDFLAGS -Wl,-rpath,$with_pmix/lib -L$with_pmix/lib"])
        ],
        [with_pmix=no])

    if test "x$with_pmix" != "xno"; then
        AC_CHECK_FILE([$with_pmix/include/pmix.h], [found pmix.h for PMIx], [AC_MSG_ERROR([could not find pmix.h for PMIx. Please use --with-pmix to point to the correct location. Configure aborted])])
        AC_CHECK_LIB([pmix], [PMIx_Init],
                     [PAC_PREPEND_FLAG([-lpmix],[LIBS])
                      PAC_PREPEND_FLAG([-Wl,-rpath,$with_pmix/lib -L$with_pmix/lib -lpmix], [WRAPPER_LIBS])
                      AC_DEFINE([USE_PMIX_API], [1], [Define if PMIx client supports PMIx])],
                     [AC_MSG_ERROR([could not find the PMIx libpmix. Please use --with-pmix to point to the correct location. Configure aborted])])
    else
        AC_CHECK_HEADER([pmix.h], [], [AC_MSG_ERROR([could not find pmix.h. Please use --with-pmix to point to the correct location. Configure aborted])])
        AC_CHECK_LIB([pmix], [PMIx_Init],
                    [PAC_PREPEND_FLAG([-lpmix],[LIBS])
                     PAC_PREPEND_FLAG([-lpmix], [WRAPPER_LIBS])
                     AC_DEFINE([USE_PMIX_API], [1], [Define if PMIx client supports PMIx])],
                    [AC_MSG_ERROR([could not find the slurm libpmix. Please use --with-pmix to point to the correct location. Configure aborted])])
    fi
else
    AC_MSG_ERROR([Selected PMI ($with_pmi) is not compatible with flux])
fi

])dnl end COND_IF

])dnl end BODY macro

[#] end of __file__
