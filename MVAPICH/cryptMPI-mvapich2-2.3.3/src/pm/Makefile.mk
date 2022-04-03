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


# util comes first, sets some variables that may be used by each process
# manager's Makefile.mk
include $(top_srcdir)/src/pm/util/Makefile.mk

include $(top_srcdir)/src/pm/gforker/Makefile.mk
include $(top_srcdir)/src/pm/remshell/Makefile.mk

## a note about DIST_SUBDIRS:
## We conditionally add DIST_SUBDIRS entries because we conditionally configure
## these subdirectories.  See the automake manual's "Unconfigured
## Subdirectories" section, which lists this rule: "Any directory listed in
## DIST_SUBDIRS and SUBDIRS must be configured."
##
## The implication for "make dist" and friends is that we should only "make
## dist" in a tree that has been configured to enable to directories that we
## want to distribute.  Because of this, we will probably need to continue using 
## the release.pl script because various SUBDIRS are incompatible with each
## other.

# has its own full automake setup, not Makefile.mk
src_pm_mpirun_src_hostfile_libhostfile_a_YFLAGS = -d -p hostfile_yy
src_pm_mpirun_src_slurm_libnodelist_a_YFLAGS = -d -p nodelist_yy
src_pm_mpirun_src_slurm_libtasklist_a_YFLAGS = -d -p tasklist_yy

if BUILD_PM_MPIRUN
include $(top_srcdir)/src/pm/mpirun/Makefile.mk
endif BUILD_PM_MPIRUN

# has its own full automake setup, not Makefile.mk
if BUILD_PM_HYDRA
SUBDIRS += src/pm/hydra
DIST_SUBDIRS += src/pm/hydra
MANDOC_SUBDIRS += src/pm/hydra
endif BUILD_PM_HYDRA
