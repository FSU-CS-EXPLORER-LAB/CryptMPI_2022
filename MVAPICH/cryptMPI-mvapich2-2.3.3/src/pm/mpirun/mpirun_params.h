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
#ifndef MPIRUN_PARAMS_H
#define MPIRUN_PARAMS_H

#include <process.h>

int read_param_file(char *paramfile, char **env);
void commandLine(int argc, char *argv[], char *totalview_cmd, char **env);
int file_exists(char *filename);
void usage(const char * arg0);

/* xxx need to add checking for string overflow, do this more carefully ... */
extern char *mpispawn_param_env;

extern int nprocs;
extern int aout_index;
extern int use_rsh;

#define ENV_LEN 1024
#define MAXLINE 1024
#define LINE_LEN 256
#define TMP_PFX "/tmp/tempfile_"
#define END     "endcmd"
#define PORT    "PARENT_ROOT_PORT_NAME"
#define ARG     "arg"
#define ENDARG  "endarg"
#define INFN    "info_num="
/*
#define END_s   strlen(END)
#define PORT_s  strlen(PORT)
#define ARG_s   strlen(ARG)
#define ENDARG_s strlen(ENDARG)
#define INFN_s  strlen(INFN)
*/

#define HOSTFILE_LEN 256
extern char hostfile[HOSTFILE_LEN + 1];

extern int xterm_on;
extern int show_on;
extern int hostfile_on;
extern int use_dirname;
extern int legacy_startup;
extern int dpm;
extern char *spawnfile;
extern char *binary_dirname;

extern int param_count;

/*
  The group active for mpispawn. NULL if no group change is required.
 */
extern char *change_group;

/* By default, use linear ssh. Enable
   -fastssh for tree based ssh */
extern int USE_LINEAR_SSH;

#endif                          /* MPIRUN_PARAMS_H */
/* vi:set sw=4 sts=4 tw=76 expandtab: */
