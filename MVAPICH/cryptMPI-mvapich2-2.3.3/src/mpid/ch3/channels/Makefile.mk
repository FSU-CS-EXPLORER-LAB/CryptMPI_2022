## -*- Mode: Makefile; -*-
## vim: set ft=automake :
##
## (C) 2011 by Argonne National Laboratory.
##     See COPYRIGHT in top-level directory.
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

include $(top_srcdir)/src/mpid/ch3/channels/nemesis/Makefile.mk
include $(top_srcdir)/src/mpid/ch3/channels/sock/Makefile.mk
include $(top_srcdir)/src/mpid/ch3/channels/mrail/Makefile.mk
include $(top_srcdir)/src/mpid/ch3/channels/psm/Makefile.mk
include $(top_srcdir)/src/mpid/ch3/channels/common/Makefile.mk
