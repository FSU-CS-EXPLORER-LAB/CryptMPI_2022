/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include "mpichinfo.h"
/* 
   Global definitions of variables that hold information about the
   version and patchlevel.  This allows easy access to the version 
   and configure information without requiring the user to run an MPI
   program 
*/
const char MPIR_Version_string[]       = MVAPICH2_VERSION;
const char MPIR_Version_date[]         = MVAPICH2_VERSION_DATE;
const char MPIR_Version_configure[]    = MVAPICH2_CONFIGURE_ARGS_CLEAN;
const char MPIR_Version_device[]       = MVAPICH2_DEVICE;
const char MPIR_Version_CC[]           = MVAPICH2_COMPILER_CC;
const char MPIR_Version_CXX[]          = MVAPICH2_COMPILER_CXX;
const char MPIR_Version_F77[]          = MVAPICH2_COMPILER_F77;
const char MPIR_Version_FC[]           = MVAPICH2_COMPILER_FC;
