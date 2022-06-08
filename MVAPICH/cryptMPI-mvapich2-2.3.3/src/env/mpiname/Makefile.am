# Copyright (c) 2001-2019, The Ohio State University. All rights
# reserved.
#
# This file is part of the MVAPICH2 software package developed by the
# team members of The Ohio State University's Network-Based Computing
# Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
#
# For detailed copyright and licensing information, please refer to the
# copyright file COPYRIGHT in the top level MVAPICH2 directory.
#

mpiname_SOURCES = mpiname.c
bin_PROGRAMS = mpiname

if BUILD_PROFILING_LIB
AM_LDFLAGS = $(top_builddir)/lib/lib@PMPILIBNAME@.la \
	     $(top_builddir)/lib/lib@MPILIBNAME@.la
else 
AM_LDFLAGS = $(top_builddir)/lib/lib@MPILIBNAME@.la
endif

