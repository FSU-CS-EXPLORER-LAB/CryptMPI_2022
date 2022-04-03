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

#include <process.h>
#include <mpirun_util.h>
#include <suffixlist.h>
#include <libnodelist_a-nodelist_parser.h>
#include <libtasklist_a-tasklist_parser.h>
#include <db/text.h>

#include <stdlib.h>
#include <stdio.h>

extern int dpm;
extern int slurm_init_nodelist (char const * const, size_t, char [][256]);
extern int slurm_init_tasklist (char const * const , size_t, size_t (*)[]);

static char const * slurm_get_nodelist()
{
    char const * env = getenv("SLURM_JOB_NODELIST");

    return env ? env : getenv("SLURM_NODELIST");
}

static int slurm_get_num_nodes()
{
    int nnodes = env2int("SLURM_JOB_NUM_NODES");

    return nnodes ? nnodes : env2int("SLURM_NNODES");
}

static char const * slurm_get_tasks_per_node()
{
    return getenv("SLURM_TASKS_PER_NODE");
}

static int slurm_fill_plist (
        int const nprocs,
        int const nnodes,
        char const * nodelist,
        char const * tasks_per_node
        )
{
    char hostname[nnodes][256];
    size_t ntasks[nnodes];
    size_t plist_index = 0, hostname_index = 0, task_index = 0;
    size_t offset = dpm ? env2int("TOTALPROCS") : 0;

    if (slurm_init_nodelist(nodelist, nnodes, hostname)) {
        return -1;
    }

    if (slurm_init_tasklist(tasks_per_node, nnodes, (size_t (*)[])ntasks)) {
        return -1;
    }

    /*
     * Fast forward offset number of processes to handle cases like dpm
     */
    while (offset >= ntasks[hostname_index]) {
        offset -= ntasks[hostname_index];
        hostname_index = (hostname_index + 1) % nnodes;
    }

    task_index += offset;

    /*
     * Now we can assign the hostnames to the plist
     */
    while (plist_index < nprocs) {
        if (task_index == ntasks[hostname_index]) {
            hostname_index = (hostname_index + 1) % nnodes;
            task_index = 0;
        }

        plist[plist_index++].hostname = db_add_text(hostname[hostname_index]);
        task_index++;
    }

    return 0;
}

int slurm_startup (int nprocs)
{
    char const * const nodelist = slurm_get_nodelist();
    int const nnodes = slurm_get_num_nodes();
    char const * const tasks_per_node = slurm_get_tasks_per_node();

    if (!(nodelist && nnodes && tasks_per_node)) {
        /*
         * SLURM JOB ID found but missing supporting variable(s)
         */
        fprintf(stderr, "Error reading SLURM environment\n");
        return -1;
    }

    return slurm_fill_plist(nprocs, nnodes, nodelist, tasks_per_node);
}

int
slurm_nprocs (void)
{
    int nprocs = env2int("SLURM_NPROCS");

    return nprocs ? nprocs : env2int("SLURM_NTASKS");
}

int check_for_slurm()
{
    char * job_id = getenv("SLURM_JOB_ID");
    char * jobid = getenv("SLURM_JOBID");

    return (job_id != NULL || jobid != NULL);
}


