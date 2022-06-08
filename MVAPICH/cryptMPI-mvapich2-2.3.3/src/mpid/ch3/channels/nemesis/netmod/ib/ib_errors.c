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
 */

#include <stdio.h>
#include <stdlib.h>

#include "upmi.h"

#include "ib_errors.h"

void ib_internal_error_abort(int line, char *file, int code, char *message)
{
  int my_rank;
  UPMI_GET_RANK(&my_rank);
  fprintf(stderr, "[%d] Abort: ", my_rank);
  fprintf(stderr, message);
  fprintf(stderr, " at line %d in file %s\n", line, file);
  fflush (stderr);
  exit(code);
}

