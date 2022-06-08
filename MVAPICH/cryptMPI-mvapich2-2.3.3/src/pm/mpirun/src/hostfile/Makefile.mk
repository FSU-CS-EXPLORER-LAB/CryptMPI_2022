##
## Copyright (c) 2001-2019, The Ohio State University. All rights
## reserved.
##
## This file is part of the MVAPICH2 software package developed by the
## team members of The Ohio State University's Network-Based Computing
## Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
##
## For detailed copyright and licensing information, please refer to the
## copyright file COPYRIGHT in the top level MVAPICH2 directory.
##

AM_CPPFLAGS += -I$(top_srcdir)/src/mpid/ch3/channels/common/include
AM_CPPFLAGS += -I$(top_srcdir)/src/pm/mpirun -I$(top_srcdir)/src/pm/mpirun/src -I$(top_srcdir)/src/pm/mpirun/src/hostfile

noinst_LIBRARIES += src/pm/mpirun/src/hostfile/libhostfile.a

src_pm_mpirun_src_hostfile_libhostfile_a_SOURCES = \
	src/pm/mpirun/src/hostfile/parser.y	\
	src/pm/mpirun/src/hostfile/scanner.l

BUILT_SOURCES += src/pm/mpirun/src/hostfile/parser.h
