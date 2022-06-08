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

#include <mpirun_util.h>
#include <string.h>

extern char **environ;
static int enabled = 0;
static int force = 0;

static size_t
get_num_of_environ_strings (void)
{
    char **ptr = environ;
    size_t num = 0;

    while (*ptr != NULL) {
        num++;
        ptr++;
    }

    return num;
}

static int
read_and_set_env (int s, size_t len)
{
    char envbuf[len];
    char *name = (char *)&envbuf, *value = (char *)&envbuf;

    if (read_socket(s, envbuf, len)) {
        return -1;
    }
    
    name = strsep(&value, "=");
    return setenv(name, value, force);
}

void
enable_send_environ (int overwrite)
{
    enabled = 1;
    force = overwrite;
}

int
send_environ (int s)
{
    size_t i, count = 0;
   
    if (enabled) {
        count = get_num_of_environ_strings();
    }

    if (write_socket(s, &count, sizeof(count))) {
        return -1;
    }

    if (count) {
        if (write_socket(s, &force, sizeof(force))) {
            return -1;
        }
    }

    for (i = 0; i < count; i++) {
        size_t len = strlen(environ[i]) + 1;

        if (write_socket(s, &len, sizeof(len))) {
            return -1;
        }

        if (write_socket(s, environ[i], len)) {
            return -1;
        }
    }

    return 0;
}

int
recv_environ (int s)
{
    size_t count;

    if (read_socket(s, &count, sizeof(count))) {
        return -1;
    }

    if (count) {
        if (read_socket(s, &force, sizeof(force))) {
            return -1;
        }

        enable_send_environ(force);
    }

    while (count--) {
        size_t len;

        if (read_socket(s, &len, sizeof(len))) {
            return -1;
        }

        if (read_and_set_env (s, len)) {
            return -1;
        }
    }

    return 0;
}

