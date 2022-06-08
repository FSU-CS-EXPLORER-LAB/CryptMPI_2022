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

#ifndef MPMD_H_
#define MPMD_H_

#include <process.h>

#include <stdio.h>
#include <string.h>

#define CONFILE_LEN 256
extern int configfile_on;
extern char configfile[CONFILE_LEN + 1];
/*
 * This structure saves the executable specification
 * written in the configuration file.
 *
 */
struct config_spec {
    struct config_spec *next;
    const char *exe;
    const char *args;
    int argc;
    int numprocs;

};

typedef struct config_spec config_spec_t;

/**
 * Push the information about an executable in the list of config_spec.
 */
void push(config_spec_t ** headRef, char *exe, char *args, int numprocs);
/*
 * Read the heterogenous config file, making sure it's proper and all the
 * executables exist.  Command-line node limits have already been applied
 * and the tasks[] list reduced accordingly, except for -numproc.
 */
process *parse_config(char *configfile, int *nprocs);
/*
 * Save the list of specification read in the configuration file in the plist used by mpirun_rsh.
 */
process *save_plist(config_spec_t * cfg_list, int nprocs);
/**
 * Add to the argument passed to mpispawn the name of the executable and its arguments.
 *
 */
char *add_argv(char *mpispawn_env, char *exe, char *args, int tmp_i);

/**
 * Count the number of arguments of an executable. In mpirun_rsh we need this information in order to send it to mpispawn.
 */
int count_args(char *args);

/**
 * Insert in the host_list the name of the executable and the arguments.
 * When mpmd is activated in spawn_one we send the list of hosts and for each host we send the
 * executable name, the number of argument and the list of argument.
 * The host_list has the following form: host1:numProc:pid1:pid2..:pidN:exe:argc:arg1:..:argN
 */
char *create_host_list_mpmd(process_groups * pglist, process * plist);
/**
 * Utility function used to extract the tokens of a string.
 */
char **tokenize(char *line, char *delim);

//void parse_host_list_mpmd(int i, int mt_nnodes, char *host_list, char **host, int** ranks, int *np, char **exe, char **argv);

#endif                          /* MPMD_H_ */
