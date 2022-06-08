## -*- Mode: Makefile; -*-
## vim: set ft=automake :
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


if BUILD_MRAIL

AM_CPPFLAGS += -I$(top_builddir)/src/mpid/ch3/channels/mrail/include	\
	       -I$(top_srcdir)/src/mpid/ch3/channels/mrail/include	\
	       -I$(top_builddir)/src/mpid/ch3/channels/common/include	\
	       -I$(top_srcdir)/src/mpid/ch3/channels/common/include	\
	       -I$(top_builddir)/src/mpid/common/locks			\
	       -I$(top_srcdir)/src/mpid/common/locks			\
	       -I$(top_builddir)/src/util/wrappers			\
	       -I$(top_srcdir)/src/util/wrappers

if BUILD_MRAIL_GEN2
AM_CPPFLAGS += -I$(top_builddir)/src/mpid/ch3/channels/mrail/src/gen2	\
	       -I$(top_srcdir)/src/mpid/ch3/channels/mrail/src/gen2
endif

include $(top_srcdir)/src/mpid/ch3/channels/mrail/src/Makefile.mk

endif
