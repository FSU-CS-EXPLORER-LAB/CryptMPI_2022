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
 *
 *  (C) 2007 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* This file creates strings for the most important configuration options.
   These are then used in the file src/mpi/init/initthread.c to initialize
   global variables that will then be included in both the library and 
   executables, providing a way to determine what version and features of
   MVAPICH2 were used with a particular library or executable. 
*/
#ifndef MPICHINFO_H_INCLUDED
#define MPICHINFO_H_INCLUDED

#define MVAPICH2_CONFIGURE_ARGS_CLEAN "--prefix=/home/sadeghil/Installed_Tools/CryptMPI --with-boringssl-include=/home/sadeghil/Installed_Tools/BoringSSL/boringssl-master/include/"
#define MVAPICH2_VERSION_DATE "Thu January 09 22:00:00 EST 2019"
#define MVAPICH2_DEVICE "ch3:mrail"
#define MVAPICH2_COMPILER_CC "gcc    -DNDEBUG -DNVALGRIND -O2"
#define MVAPICH2_COMPILER_CXX "g++   -DNDEBUG -DNVALGRIND -O2"
#define MVAPICH2_COMPILER_F77 "gfortran -L/lib -L/lib   -O2"
#define MVAPICH2_COMPILER_FC "gfortran   -O2"

#endif
