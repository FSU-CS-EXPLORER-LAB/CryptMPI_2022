#ifndef SPAWN_INFO_H
#define SPAWN_INFO_H 1
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

#define MAXLINE 1024

typedef struct arg {
    char const * arg;
    struct arg * next;
} arg_list;

typedef struct spawn_info {
    arg_list command;
    int nprocs;
    char const * wdir;
    char const * path;
    char const * port;
    struct spawn_info * next;

    int totspawns;
    int spawnsdone;
    int dpmtot;
    int dpmindex;
    int launch_num;
    char buf[MAXLINE];
    char linebuf[MAXLINE];
    char runbuf[MAXLINE];
    char argbuf[MAXLINE];
    char *spawnfile;
} spawn_info_t;

#endif
