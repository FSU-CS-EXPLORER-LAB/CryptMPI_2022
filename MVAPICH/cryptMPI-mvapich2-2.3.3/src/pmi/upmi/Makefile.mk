## -*- Mode: Makefile; -*-
## vim: set ft=automake :
##
## (C) 2011 by Argonne National Laboratory.
##     See COPYRIGHT in top-level directory.
##

#if BUILD_PMI_SIMPLE

mpi_core_sources += \
    src/pmi/upmi/upmi.c

AM_CPPFLAGS += -I$(top_srcdir)/src/pmi/simple

#endif BUILD_PMI_SIMPLE

