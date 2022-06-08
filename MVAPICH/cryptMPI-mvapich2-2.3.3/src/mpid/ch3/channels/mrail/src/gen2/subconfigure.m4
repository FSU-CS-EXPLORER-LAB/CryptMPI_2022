[#] start of __file__
dnl MPICH2_SUBCFG_AFTER=src/mpid/ch3/channels/mrail

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX, [
    AS_IF([test "$build_mrail" = yes -a "x$with_rdma" = xgen2],
	  [build_mrail_gen2=yes],
	  [build_mrail_gen2=no])
    AM_CONDITIONAL([BUILD_MRAIL_GEN2], [test "$build_mrail_gen2" = yes])
    AM_COND_IF([BUILD_MRAIL_GEN2], [
	AC_MSG_NOTICE([RUNNING PREREQ FOR ch3:mrail:gen2])
    ])dnl end AM_COND_IF(BUILD_MRAIL_GEN2,...)
])dnl
dnl
dnl _BODY handles the former role of configure in the subsystem
AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX, [
    AM_COND_IF([BUILD_MRAIL_GEN2], [
	AC_MSG_NOTICE([RUNNING CONFIGURE FOR ch3:mrail:gen2])
        AC_DEFINE([MRAIL_GEN2_INTERFACE], [1],
            [Define to enable GEN2 interface])
    ])dnl end AM_COND_IF(BUILD_MRAIL_GEN2,...)
])dnl end _BODY
[#] end of __file__
