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

#include <src/pm/mpirun/mpirun_rsh.h>
#include <spawn_info.h>
#include <mpirun_params.h>
#include <string.h>

extern char * spawnfile;
extern spawn_info_t spinf;

static void
append_arg (arg_list * command, char const * arg)
{
    static arg_list * last = NULL;
    static arg_list * cache = NULL;

    if (cache != command) {
        cache = command;
        last = command;

        while (last->next) {
            last = last->next;
        }
    }

    last->next = malloc(sizeof(struct arg));
    last = last->next;
    last->arg = arg;
    last->next = NULL;
}

static void
get_line (void * ptr, char * fill, int is_file)
{
    int i = 0, ch;
    FILE *fp;
    char *buf;

    if (is_file) {
        fp = ptr;
        while ((ch = fgetc(fp)) != '\n') {
            fill[i] = ch;
            ++i;
        }
    } else {
        buf = ptr;
        while (buf[i] != '\n') {
            fill[i] = buf[i];
            ++i;
        }
    }
    fill[i] = '\0';
}

static void
store_info (spawn_info_t * si, char const * key, char const * val)
{
    if (0 == (strcmp(key, "wdir"))) {
        si->wdir = val;
    }

    else if (0 == (strcmp(key, "path"))) {
        si->path = val;
    }
}

static void
read_infn (spawn_info_t * si, FILE * fp, int num_pairs)
{
    while (num_pairs--) {
        char key[1024];
        char value[1024];

        get_line(fp, key, 1);
        get_line(fp, value, 1);

        store_info(si, strdup(key), strdup(value));
    }
}

/**
 * Read DPM specfile and return the number of spawn info list.
 */
extern spawn_info_t *
read_dpm_specfile (int n_spawns)
{
    spawn_info_t * rv = NULL, * si = NULL;
    FILE * fp = fopen(spawnfile, "r");
    int i;
    char buffer[1024];


    if (!fp) {
        fprintf(stderr, "spawn specification file not found\n");
        return NULL;
    }

    for (i = 0; i < n_spawns; i++) {
        int done = 0;

        if (!si) {
            rv = si = malloc(sizeof(spawn_info_t));
        }

        else {
            si->next = malloc(sizeof(spawn_info_t));
            si = si->next;
        }

        if (!si) {
            return NULL;
        }

        memset(si, 0, sizeof(spawn_info_t));

        get_line(fp, buffer, 1);
        si->nprocs = atoi(buffer);

        get_line(fp, buffer, 1);
        si->command.arg = strdup(buffer);
        si->command.next = NULL;

        do {
            get_line(fp, buffer, 1);

            if (0 == (strncmp(buffer, ARG, strlen(ARG)))) {
                append_arg(&(si->command), strdup(index(buffer, '=') + 1));
            }

            else if (0 == (strncmp(buffer, PORT, strlen(PORT)))) {
                get_line(fp, buffer, 1);
                si->port = mkstr("%s='%s'", PORT, buffer);
            }

            else if (0 == (strncmp(buffer, INFN, strlen(INFN)))) {
                read_infn(si, fp, atoi(index(buffer, '=') + 1));
            }

            else if (0 == (strncmp(buffer, END, strlen(END)))) {
                done = 1;
            }
        } while (!done);
    }

    return rv;
}

/* vi:set sw=4 sts=4 tw=80 expandtab: */
