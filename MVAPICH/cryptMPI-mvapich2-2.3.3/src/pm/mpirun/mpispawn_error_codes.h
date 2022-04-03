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

#ifndef MPISPAWN_ERROR_CODES_H
#define MPISPAWN_ERROR_CODES_H 1

typedef enum {
    MPISPAWN_MPIPROCESS_ERROR = 1,         // An MPI process got an error
    MPISPAWN_MPIPROCESS_NONZEROEXIT = 2,   // An MPI process returned a non-zer
    MPISPAWN_DPM_REQ = 3,                  // DPM request
    MPISPAWN_PMI_READ_ERROR = 4,           // MPISPAWN got an error while readi
    MPISPAWN_PMI_WRITE_ERROR = 5,          // MPISPAWN got an error while writi
    MPISPAWN_INTERNAL_ERROR = 6,           // MPISPAWN got an internal error
    MPISPAWN_CLEANUP_SIGNAL = 7,           // MPISPAWN received a cleanup signa
    MPISPAWN_TRIGGER_MIGRATION = 8,        // MPISPAWN triggers a migration
} mpispawn_error_code;

static inline const char *
get_mpispawn_error_str (mpispawn_error_code ec)
{
    switch( ec ) {
        case MPISPAWN_MPIPROCESS_ERROR:
            return "MPI process error";
        case MPISPAWN_MPIPROCESS_NONZEROEXIT:
            return "An MPI process returned a non-zero exit code";
        case MPISPAWN_DPM_REQ:
            return "DPM request (not an error)";
        case MPISPAWN_PMI_READ_ERROR:
            return "Error while reading a PMI socket";
        case MPISPAWN_PMI_WRITE_ERROR:
            return "Error while writing a PMI socket";
        case MPISPAWN_INTERNAL_ERROR:
            return "MPISPAWN internal error";
        case MPISPAWN_CLEANUP_SIGNAL:
            return "MPISPAWN got cleanup signal";
        case MPISPAWN_TRIGGER_MIGRATION:
            return "MPISPAWN triggers a migration (not an error)";
        default:
            return "Unknown error";
    }
}

#endif
