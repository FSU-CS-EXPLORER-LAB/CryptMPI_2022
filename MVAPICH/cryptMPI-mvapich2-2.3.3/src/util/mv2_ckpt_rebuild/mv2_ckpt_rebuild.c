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

#include "mpi.h"
#include <stdio.h>

int main(int argc,char *argv[])
{
    int myid, numprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    /* Invoke SCR_Init() in order to rebuild any lost checkpoints */
    if (!myid)
        fprintf(stderr, "Reinitializing SCR to verify integrity of cached checkpoints.\n");
    SCR_Init();
    if (!myid)
        fprintf(stderr, "Checkpoints are healthy. Restart the application using the \"cr_restart\" utility.\n");

    /* Make sure that MPI_Finalize() does not implicitly trigger SCR_Finalize()
     * when rebuilding checkpoints. If it does, then SCR considers the job to
     * have completed successfully and prevents a restart */
    SCR_Donot_Finalize();
    MPI_Finalize();

    /* After rebuilding the checkpoints, toggle back the implicit SCR_Finalize()
     * to allow the restarted MPI application to finalize cleanly */
    SCR_Do_Finalize();
    return 0;
}
