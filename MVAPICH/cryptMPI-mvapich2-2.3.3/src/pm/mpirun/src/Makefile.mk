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

AM_CPPFLAGS += -I$(srcdir)
 
include $(top_srcdir)/src/pm/mpirun/src/db/Makefile.mk
include $(top_srcdir)/src/pm/mpirun/src/hostfile/Makefile.mk
include $(top_srcdir)/src/pm/mpirun/src/slurm/Makefile.mk
include $(top_srcdir)/src/pm/mpirun/src/pbs/Makefile.mk
