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

#ifndef _MY_CRFS_
#define _MY_CRFS_

#include <semaphore.h>
#include "atomic.h"
#include "common_crfs.h"

enum {
    /// crfs-mode
    MODE_WRITEAGGRE = 1,        // CRFS works in Write-Aggre mode
    MODE_MIG = 2,               // CRFS works in Proc-Mig mode

    // mig-role
    ROLE_INVAL = 0,             // invalid role...
    ROLE_MIG_SRC = 1,           // In MIG mode: at client side (mig src )
    ROLE_MIG_TGT = 2,           // In MIG mode: at srv-side ( mig target)

    CRFS_CLIENT = 1,            // at cli-side: only write-aggregation
    CRFS_SRV = 2,               // at srv-side: proc-migration, use in-mem mig mode
};

#define MAX_HOSTNAME_LENGTH    64
/**
this struct contains all infor needed to do a migration.
**/
typedef struct mig_info_s {
    /// input::
    char src[MAX_HOSTNAME_LENGTH];
    char tgt[MAX_HOSTNAME_LENGTH];
    int port;

    sem_t sem;                  // to sync between CRFS & mig-thread

    atomic_t chunk_cnt;         // num of chunks to be RRed to server
    sem_t chunk_comp_sem;       // when all chunks have been RRed to server, post a sem

    /// out:
    struct ib_connection *conn;
    int fail_flag;

} mig_info_t;

extern int crfs_mode;
extern int mig_role;
extern char crfs_mig_filename[128];

int start_crfs(char *sessionid, char *fullpath, int mig);
int stop_crfs(const char *crfs_mnt, int crfs_pid);
int stop_crfs_mig();
int stop_crfs_wa();
int crfs_start_mig(char *tgt);
int crfs_stop_mig();
int stop_crfs_wa();
int stop_crfs_mig();

int crfs_init_imp(int mode);
int crfs_destroy_imp(int mode);

int ibcli_start_mig(mig_info_t * minfo);
void ibcli_end_mig(mig_info_t * minfo);

int ibcli_main_entry(void *p);
void ibcli_main_exit();
int ibsrv_main_entry(void *p);
int ibsrv_main_exit();

#endif                          // _my_crfs_
