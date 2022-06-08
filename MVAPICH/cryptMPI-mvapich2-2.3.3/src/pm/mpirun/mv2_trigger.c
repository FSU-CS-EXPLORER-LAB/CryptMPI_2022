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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpichconf.h"

#ifdef CR_FTB

#include <libftb.h>

#define FTB_MAX_SUBSCRIPTION_STR 64

#define FTB_MIGRATE_EVENT_INFO {            \
        {"FTB_MIGRATE_TRIGGER", "info"}, \
}

/* Index into the Event Info Table */
#define FTB_MIGRATE_TRIGGER 0

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
#define EVENT(n) (mig_ftb_events[n].event_name)

static FTB_client_t ftb_cinfo;
static FTB_event_info_t mig_ftb_events[] = FTB_MIGRATE_EVENT_INFO;
static FTB_client_handle_t ftb_handle;

int ftb_init_done = 0;

static int mig_ftb_init();
static void mig_ftb_finalize();

static int mig_ftb_init()
{
    int ret;

    memset(&ftb_cinfo, 0, sizeof(ftb_cinfo));
    strcpy(ftb_cinfo.client_schema_ver, "0.5");
    strcpy(ftb_cinfo.event_space, "FTB.MPI.MIG_TRIGGER");
    strcpy(ftb_cinfo.client_name, "MV2_MIGRATE");

    /* sessionid should be <= 16 bytes since client_jobid is 16 bytes. */
    snprintf(ftb_cinfo.client_jobid, FTB_MAX_CLIENT_JOBID, "%d", 16);   //sessionid

    strcpy(ftb_cinfo.client_subscription_style, "FTB_SUBSCRIPTION_BOTH");
    ftb_cinfo.client_polling_queue_len = 10;    //nprocs length

    ret = FTB_Connect(&ftb_cinfo, &ftb_handle);
    if (ret != FTB_SUCCESS)
        goto err_connect;

    ret = FTB_Declare_publishable_events(ftb_handle, NULL, mig_ftb_events, 1);  //max events in mig_ftb_events
    if (ret != FTB_SUCCESS)
        goto err_declare_events;

    ftb_init_done = 1;
    return (0);

  err_connect:
    fprintf(stderr, "FTB_Connect() in MV2_MIGRATE failed with %d\n", ret);
    ret = -1;
    goto exit_connect;

  err_declare_events:
    fprintf(stderr, "FTB_Declare_publishable_events() in MV2_MIGRATE failed with %d\n", ret);
    ret = -2;
    goto exit_declare_events;

  exit_declare_events:
    FTB_Disconnect(ftb_handle);
  exit_connect:
    return (ret);
}

static void mig_ftb_finalize()
{
    if (ftb_init_done) {
        usleep(10000);
        FTB_Disconnect(ftb_handle);
    }
    ftb_init_done = 0;
}

int main(int argc, char *argv[])
{
    int ret;

    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;

    if (argc < 2) {
        printf("Usage:  %s [failing node]\n", argv[0]);
        exit(-1);
    }
    /// connect to FTB
    mig_ftb_init();

    // publish event
    SET_EVENT(eprop, FTB_EVENT_NORMAL, argv[1]);    //"Sample:Payload");
    fprintf(stderr, "Event has been sent\n");
    ret = FTB_Publish(ftb_handle, EVENT(FTB_MIGRATE_TRIGGER), &eprop, &ehandle);
    if (ret == 0) {
        fprintf(stderr, "Predict %s will fail soon...\nHas triggered an event...\n", argv[1]);
    } else if (ret != 0) {
        fprintf(stderr, "FTB_MIG Pub() failed with %d\n", ret);
    }
    //// finalize FTB
    mig_ftb_finalize();
    return 0;
}

#else                           /* CR_FTB undefined */

int main(int argc, char *argv[])
{
    fprintf(stderr, "Error: please enable FTB to use mv2_trigger\n");
    return 1;
}

#endif                          /* CR_FTB */
