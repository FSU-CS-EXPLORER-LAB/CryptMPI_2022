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

// There are a couple of symbols that need to be #defined before
// #including all the headers.

#ifndef _FUSE_PARAMS_H_
#define _FUSE_PARAMS_H_

// The FUSE API has been changed a number of times.  So, our code
// needs to define the version of the API that we assume.  As of this
// writing, the most current API version is 26
#define FUSE_USE_VERSION 26

// need this to get pwrite().  I have to use setvbuf() instead of
// setlinebuf() later in consequence.
#define _XOPEN_SOURCE 500

// maintain crfs state in here
#include <limits.h>
#include <stdio.h>

#define MAX_PATH_LEN    (256)

struct crfs_state {
    FILE *logfile;
    char *rootdir;
};

#define CRFS_DATA ( (struct crfs_state *)(fuse_get_context()->private_data) )

#endif                          // end of _FUSE_PARAMS_H_
