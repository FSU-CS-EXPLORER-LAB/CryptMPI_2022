#ifndef PROCESS_H
#define PROCESS_H 1
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

#include <unistd.h>
#include <sys/types.h>
#include <spawn_info.h>

typedef enum {
    P_NOTSTARTED,
    P_STARTED,
    P_CONNECTED,
    P_DISCONNECTED,
    P_RUNNING,
    P_FINISHED,
    P_EXITED
} process_state;

typedef struct {
    char const * hostname;
    char const * device;
    pid_t pid;
    pid_t remote_pid;
    int port;
    int control_socket;
    process_state state;
    //These information are about the mpmd (multi process multiple data) option
    char *executable_name;
    char *executable_args;
    int argc;
} process;

typedef struct {
    const char *hostname;
    pid_t pid;
    int *plist_indices;
    size_t npids, npids_allocated;
    spawn_info_t * si;

    pid_t local_pid;            //the local forked() proc pid
} process_group;

typedef struct {
    process_group *data;
    process_group **index;
    size_t npgs, npgs_allocated;
} process_groups;

extern process * plist;
extern process_groups * pglist;

/* vi:set sw=4 sts=4 tw=80: */
#endif
