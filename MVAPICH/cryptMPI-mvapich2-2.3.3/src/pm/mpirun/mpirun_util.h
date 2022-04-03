#ifndef MPIRUN_UTIL_H
#define MPIRUN_UTIL_H
/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>

char *vedit_str(char *const, const char *, va_list);
char *edit_str(char *const, char const *const, ...);
char *mkstr(const char *, ...);
char *append_str(char *, char const *const);

int read_socket(int, void *, size_t);
int write_socket(int, void *, size_t);
int connect_socket(char *, char *);

typedef struct _process_info {
    /* pid_t pid; */
    long pid;
    int rank;
} process_info_t;

struct MPIR_PROCDESC {
    char const * host_name;
    char const * executable_name;
    long pid;
};

struct spawn_info_s {
    char spawnhost[32];
    int sparenode;
};
#define MPIR_PROCDESC_s (sizeof (struct MPIR_PROCDESC))
#define process_info_s (sizeof (process_info_t))

static inline int env2int(char *env_ptr)
{
    return (env_ptr = getenv(env_ptr)) ? atoi(env_ptr) : 0;
}

static inline char *env2str(char *env_ptr)
{
    return (env_ptr = getenv(env_ptr)) ? strdup(env_ptr) : NULL;
}

#endif
