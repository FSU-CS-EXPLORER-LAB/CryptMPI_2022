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

mpi_core_sources += \
	src/mpid/ch3/channels/common/src/util/crc32h.c  \
	src/mpid/ch3/channels/common/src/util/mv2_config.c  \
	src/mpid/ch3/channels/common/src/util/error_handling.c  \
	src/mpid/ch3/channels/common/src/util/debug_utils.c  \
	src/mpid/ch3/channels/common/src/util/mv2_clock.c
