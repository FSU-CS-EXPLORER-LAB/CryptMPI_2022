/*
 * Copyright (c) 2009, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-411039.
 * All rights reserved.
 * This file is part of The Scalable Checkpoint / Restart (SCR) library.
 * For details, see https://sourceforge.net/projects/scalablecr/
 * Please also read this file: LICENSE.TXT.
*/

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

/* enable C++ codes to include this header directly */
#ifdef __cplusplus
extern "C" {
#endif

/* constants returned from SCR functions for success and failure */
#define SCR_SUCCESS (0)

/* maximum characters in a filename returned by SCR */
#define SCR_MAX_FILENAME 1024

/* see the SCR user manual for full details on these functions */

/* initialize the SCR library */
int SCR_Init();

/* shut down the SCR library */
int SCR_Finalize();
int SCR_Donot_Finalize();
int SCR_Do_Finalize();

/* determine whether a checkpoint should be taken at the current time */
int SCR_Need_checkpoint(int* flag);

/* inform library that a new checkpoint is starting */
int SCR_Start_checkpoint();

/* determine the path and filename to be used to open a file */
int SCR_Route_file(const char* name, char* file);

/* inform library that the current checkpoint is complete */
int SCR_Complete_checkpoint(int valid);

/* enable C++ codes to include this header directly */
#ifdef __cplusplus
} /* extern "C" */
#endif
