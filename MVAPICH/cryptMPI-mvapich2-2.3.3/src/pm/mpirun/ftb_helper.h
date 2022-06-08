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

#ifndef FTB_HELPER_H
#define FTB_HELPER_H

#include "mpichconf.h"

#ifdef CR_FTB

#include <libftb.h>

#define FTB_MAX_SUBSCRIPTION_STR 128

/////////////////////////////////////////////////////////
    // max-event-name-len=32,  max-severity-len=16
#define CR_FTB_EVENT_INFO {               \
        {"CR_FTB_CHECKPOINT",    "info"}, \
        {"CR_FTB_MIGRATE",       "info"}, \
        {"CR_FTB_MIGRATE_PIIC",  "info"}, \
        {"CR_FTB_CKPT_DONE",     "info"}, \
        {"CR_FTB_CKPT_FAIL",     "info"}, \
        {"CR_FTB_RSRT_DONE",     "info"}, \
        {"CR_FTB_RSRT_FAIL",     "info"}, \
        {"CR_FTB_APP_CKPT_REQ",  "info"}, \
        {"CR_FTB_CKPT_FINALIZE", "info"}, \
        {"CR_FTB_MIGRATE_PIC",   "info"}, \
        {"FTB_MIGRATE_TRIGGER",  "info"},  \
        {"MPI_PROCS_CKPTED", "info"},       \
        {"MPI_PROCS_CKPT_FAIL", "info"},    \
        {"MPI_PROCS_RESTARTED", "info"},    \
        {"MPI_PROCS_RESTART_FAIL", "info"}, \
        {"MPI_PROCS_MIGRATED", "info"},     \
        {"MPI_PROCS_MIGRATE_FAIL", "info"} \
}

    // Index into the Event Info Table
#define CR_FTB_CHECKPOINT    0
#define CR_FTB_MIGRATE       1
#define CR_FTB_MIGRATE_PIIC  2
#define CR_FTB_CKPT_DONE     3
#define CR_FTB_CKPT_FAIL     4
#define CR_FTB_RSRT_DONE     5
#define CR_FTB_RSRT_FAIL     6
#define CR_FTB_APP_CKPT_REQ  7
#define CR_FTB_CKPT_FINALIZE 8
#define CR_FTB_MIGRATE_PIC   9
#define FTB_MIGRATE_TRIGGER  10
    // start of standard FTB MPI events
#define MPI_PROCS_CKPTED        11
#define MPI_PROCS_CKPT_FAIL     12
#define MPI_PROCS_RESTARTED     13
#define MPI_PROCS_RESTART_FAIL  14
#define MPI_PROCS_MIGRATED      15
#define MPI_PROCS_MIGRATE_FAIL 16

#define CR_FTB_EVENTS_MAX    17
////////////////////////////////////////////////////

/* Type of event to throw */
#define FTB_EVENT_NORMAL   1
#define FTB_EVENT_RESPONSE 2

/* Macro to initialize the event property structure */
#define SET_EVENT(_eProp, _etype, _payload...)             \
do {                                                       \
    _eProp.event_type = _etype;                            \
    snprintf(_eProp.event_payload, FTB_MAX_PAYLOAD_DATA,   \
                _payload);                                 \
} while(0)

/* Macro to pick an CR_FTB event */
#define EVENT(n) (cr_ftb_events[n].event_name)

#endif                          /* CR_FTB */

#endif                          /* FTB_HELPER_H */
