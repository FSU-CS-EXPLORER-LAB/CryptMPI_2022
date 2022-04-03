/*RAM
 * Copyright (C) 1999-2001 The Regents of the University of California
 * (through E.O. Lawrence Berkeley National Laboratory), subject to
 * approval by the U.S. Department of Energy.
 *
 * Use of this software is under license. The license agreement is included
 * in the file MVICH_LICENSE.TXT.
 *
 * Developed at Berkeley Lab as part of MVICH.
 *
 * Authors: Bill Saphir      <wcsaphir@lbl.gov>
 *          Michael Welcome  <mlwelcome@lbl.gov>
 */

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

#include <mpichconf.h>
#include <src/pm/mpirun/mpirun_rsh.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <math.h>
#include "mpispawn_tree.h"
#include "mpirun_util.h"
#include "mpmd.h"
#include "mpirun_dbg.h"
#include "mpirun_params.h"
#include <mpirun_environ.h>
#include "mpirun_ckpt.h"

#if defined(_NSIG)
#define NSIG _NSIG
#endif                          /* defined(_NSIG) */

extern int read_hostfile(char *hostfile_name);

process *plist = NULL;
int nprocs = 0;
int aout_index = 0;

/* xxx need to add checking for string overflow, do this more carefully ... */
char *mpispawn_param_env = NULL;
char *spawnfile;
char *binary_dirname;

#if defined(USE_RSH)
int use_rsh = 1;
#else                           /* defined(USE_RSH) */
int use_rsh = 0;
#endif                          /* defined(USE_RSH) */

int xterm_on = 0;
int show_on = 0;
int use_dirname = 1;
int hostfile_on = 0;
int param_count = 0;
int legacy_startup = 0;
int dpm = 0;
extern spawn_info_t spinf;
int USE_LINEAR_SSH = 1;         /* By default, use linear ssh. Enable
                                   -fastssh for tree based ssh */

char hostfile[HOSTFILE_LEN + 1];

/*
  The group active for mpispawn. NULL if no group change is required.
 */
char *change_group = NULL;

static struct option option_table[] = {
    {"np", required_argument, 0, 0},    // 0
    {"debug", no_argument, 0, 0},
    {"xterm", no_argument, 0, 0},
    {"hostfile", required_argument, 0, 0},
    {"show", no_argument, 0, 0},    // 5
    {"rsh", no_argument, 0, 0},
    {"ssh", no_argument, 0, 0},
    {"help", no_argument, 0, 0},
    {"v", no_argument, 0, 0},
    {"tv", no_argument, 0, 0},  // 10
    {"legacy", no_argument, 0, 0},
    {"startedByTv", no_argument, 0, 0},
    {"spawnfile", required_argument, 0, 0},
    {"dpm", no_argument, 0, 0},
    {"fastssh", no_argument, 0, 0}, // 15
    //This option is to activate the mpmd, it requires the configuration file as argument
    {"config", required_argument, 0, 0},
    {"dpmspawn", required_argument, 0, 0},
    // This option enables the group selection for mpispawns
    {"sg", required_argument, 0, 0},
    {"export", no_argument, 0, 0},
    {"export-all", no_argument, 0, 0},
#if defined(CKPT) && defined(CR_FTB)
    {"sparehosts", required_argument, 0, 0},    // 20
#endif
    {0, 0, 0, 0}
};

#if !defined(HAVE_GET_CURRENT_DIR_NAME)
char *get_current_dir_name()
{
    struct stat64 dotstat;
    struct stat64 pwdstat;
    char *pwd = getenv("PWD");

    if (pwd != NULL && stat64(".", &dotstat) == 0 && stat64(pwd, &pwdstat) == 0 && pwdstat.st_dev == dotstat.st_dev && pwdstat.st_ino == dotstat.st_ino) {
        /* The PWD value is correct. */
        return strdup(pwd);
    }

    size_t size = 1;
    char *buffer;

    for (;; ++size) {
        buffer = malloc(size);

        if (!buffer) {
            return NULL;
        }

        if (getcwd(buffer, size) == buffer) {
            break;
        }

        free(buffer);

        if (errno != ERANGE) {
            return NULL;
        }
    }

    return buffer;
}
#endif                          /* !defined(HAVE_GET_CURRENT_DIR_NAME) */

#if !defined(HAVE_STRNDUP)
char *strndup(const char *s, size_t n)
{
    size_t len = strlen(s);

    if (n < len) {
        len = n;
    }

    char *result = malloc(len + 1);

    if (!result) {
        return NULL;
    }

    result[len] = '\0';
    return memcpy(result, s, len);
}
#endif                          /* !defined(HAVE_STRNDUP) */

#define PARAMFILE_LEN 256

/**
 * Command line analysis function.
 *
 * mpirun [-debug] [-xterm] -np N [-hostfile hfile | h1 h2 h3 ... hN] a.out [args]
 */
void commandLine(int argc, char *argv[], char *totalview_cmd, char **env)
{
    int i;
    int c, option_index;

    do {
        c = getopt_long_only(argc, argv, "+", option_table, &option_index);
        switch (c) {
        case '?':
        case ':':
            usage(argv[0]);
            exit(EXIT_FAILURE);
            break;
        case EOF:
            break;
        case 0:
            switch (option_index) {
            case 0:            /* -np */
                nprocs = atoi(optarg);
                if (nprocs < 1) {
                    usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                break;
            case 1:            /* -debug */
                debug_on = 1;
                xterm_on = 1;
                break;
            case 2:            /* -xterm */
                xterm_on = 1;
                break;
            case 3:            /* -hostfile */
                hostfile_on = 1;
                strncpy(hostfile, optarg, HOSTFILE_LEN);
                if (strlen(optarg) >= HOSTFILE_LEN - 1)
                    hostfile[HOSTFILE_LEN] = '\0';
                break;
            case 4:
                show_on = 1;
                break;
            case 5:
                use_rsh = 1;
                break;
            case 6:
                use_rsh = 0;
                break;
            case 7:
                usage(argv[0]);
                exit(EXIT_SUCCESS);
                break;
            case 8:
                PRINT_MVAPICH2_VERSION();
                exit(EXIT_SUCCESS);
                break;
            case 9:
                {
                    /* -tv */
                    char *tv_env;
                    int count, idx;
                    char **new_argv;
                    tv_env = getenv("TOTALVIEW");
                    if (tv_env != NULL) {
                        strncpy(totalview_cmd, tv_env, TOTALVIEW_CMD_LEN);
                    } else {
                        fprintf(stderr, "TOTALVIEW env is NULL, use default: %s\n", TOTALVIEW_CMD);
                        sprintf(totalview_cmd, "%s", TOTALVIEW_CMD);
                    }
                    new_argv = (char **) malloc(sizeof(char **) * argc + 3);
                    new_argv[0] = totalview_cmd;
                    new_argv[1] = argv[0];
                    new_argv[2] = "-a";
                    new_argv[3] = "-startedByTv";
                    idx = 4;
                    for (count = 1; count < argc; count++) {
                        if (strcmp(argv[count], "-tv"))
                            new_argv[idx++] = argv[count];
                    }
                    new_argv[idx] = NULL;
                    if (execv(new_argv[0], new_argv)) {
                        perror("execv");
                        exit(EXIT_FAILURE);
                    }

                }
                break;
            case 10:
                legacy_startup = 1;
                break;
            case 11:
                /* -startedByTv */
                use_totalview = 1;
                debug_on = 1;
                break;
            case 12:           /* spawnspec given */
                spawnfile = strdup(optarg);
                DBG(fprintf(stderr, "spawn spec file = %s\n", spawnfile));
                break;
            case 13:
                dpm = 1;
                break;
            case 14:           /* -fastssh */
#if !defined(CR_FTB)
                /* disable hierarchical SSH if migration is enabled */
                USE_LINEAR_SSH = 0;
#endif 
                break;
                //With this option the user want to activate the mpmd
            case 15:
                configfile_on = 1;
                strncpy(configfile, optarg, CONFILE_LEN);
                if (strlen(optarg) >= CONFILE_LEN - 1)
                    configfile[CONFILE_LEN] = '\0';
                break;
            case 16:
                spinf.totspawns = atoi(optarg);
                break;
            case 17:
                /* sg: change the active group */
                change_group = optarg;
                DBG(printf("Group change requested: '%s'\n", change_group));
                break;
            case 18:
                enable_send_environ(0);
                break;
            case 19:
                enable_send_environ(1);
                break;
#if defined(CKPT) && defined(CR_FTB)
            case 20:
                sparehosts_on = 1;
                strncpy(sparehostfile, optarg, HOSTFILE_LEN);
                if (strlen(optarg) >= HOSTFILE_LEN - 1) {
                    sparehostfile[HOSTFILE_LEN] = 0;
                }
                break;
#endif
            default:
                fprintf(stderr, "Unknown option\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
                break;
            }
            break;
        default:
            fprintf(stderr, "Unreachable statement!\n");
            usage(argv[0]);
            exit(EXIT_FAILURE);
            break;
        }
    }
    while (c != EOF);

    if (!nprocs && !configfile_on) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    binary_dirname = dirname(strdup(argv[0]));
    if (strlen(binary_dirname) == 1 && argv[0][0] != '.') {
        use_dirname = 0;
    }
    //If the mpmd is active we need to parse the configuration file
    if (configfile_on) {
        /*TODO In the future the user can add the nprocs on the command line. Now the
         * number of processes is defined in the configfile */
        nprocs = 0;
        plist = parse_config(configfile, &nprocs);
        DBG(fprintf(stderr, "PARSED CONFIG FILE\n"));

    }

    if (!hostfile_on) {
        /* get hostnames from argument list */
        if (strchr(argv[optind], '=') || argc - optind < nprocs + 1) {
            sprintf(hostfile, "%s/.mpirun_hosts", env2str("HOME"));
            if (file_exists(hostfile)) {
                hostfile_on = 1;
                aout_index = optind;
                goto cont;
            } else {
                fprintf(stderr, "Without hostfile option, hostnames must be " "specified on command line.\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        }
        aout_index = nprocs + optind;
    } else {                    /* if (!hostfile_on) */

        aout_index = optind;
    }

  cont:
    if (!configfile_on) {
        plist = malloc(nprocs * sizeof(process));
        if (plist == NULL) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < nprocs; i++) {
            plist[i].state = P_NOTSTARTED;
            plist[i].device = NULL;
            plist[i].port = -1;
            plist[i].remote_pid = 0;
            //TODO ADD EXECNAME AND ARGS

        }
    }

    /* grab hosts from command line or file */
    if (hostfile_on) {
        read_hostfile(hostfile);
    } else {
        for (i = 0; i < nprocs; i++) {
            plist[i].hostname = argv[optind + i];
        }
    }
}

void usage(const char * arg0)
{
    fprintf(stderr, "usage: mpirun_rsh [-v] [-sg group] [-rsh|-ssh] "
            "[-debug] -[tv] [-xterm] [-show] [-legacy] [-export|-export-all] "
            "-np N "
#if defined(CKPT) && defined(CR_FTB)
            "[-sparehosts sparehosts_file] "
#endif
            "(-hostfile hfile | h1 h2 ... hN) a.out args | -config configfile (-hostfile hfile | h1 h2 ... hN)]\n");
    fprintf(stderr, "Where:\n");
    fprintf(stderr, "\tsg         => " "execute the processes as different group ID\n");
    fprintf(stderr, "\trsh        => " "to use rsh for connecting\n");
    fprintf(stderr, "\tssh        => " "to use ssh for connecting\n");
    fprintf(stderr, "\tdebug      => " "run each process under the control of gdb\n");
    fprintf(stderr, "\ttv         => " "run each process under the control of totalview\n");
    fprintf(stderr, "\txterm      => " "run remote processes under xterm\n");
    fprintf(stderr, "\tshow       => " "show command for remote execution but don't run it\n");
    fprintf(stderr, "\tlegacy     => " "use old startup method (1 ssh/process)\n");
    fprintf(stderr, "\texport     => " "automatically export environment to remote processes\n");
    fprintf(stderr, "\texport-all => " "automatically export environment to remote processes even if already set remotely\n");
    fprintf(stderr, "\tnp         => " "specify the number of processes\n");
    fprintf(stderr, "\th1 h2...   => " "names of hosts where processes should run\n");
    fprintf(stderr, "or\thostfile   => " "name of file containing hosts, one per line\n");
    fprintf(stderr, "\ta.out      => " "name of MPI binary\n");
    fprintf(stderr, "\targs       => " "arguments for MPI binary\n");
    fprintf(stderr, "\tconfig     => " "name of file containing the exe information: each line has the form -n numProc : exe args\n");
#if defined(CKPT) && defined(CR_FTB)
    fprintf(stderr, "\tsparehosts => " "file containing the spare hosts for migration, one per line\n");
#endif
    fprintf(stderr, "\n");
}

int file_exists(char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp) {
        fclose(fp);
        return 1;
    }
    return 0;
}

/* vi:set sw=4 sts=4 tw=76 expandtab: */
