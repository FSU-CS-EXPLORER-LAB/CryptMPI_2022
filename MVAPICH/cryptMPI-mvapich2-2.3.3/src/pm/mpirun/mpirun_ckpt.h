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
#ifndef _MPIRUN_CKPT_H
#define _MPIRUN_CKPT_H

#include "mpichconf.h"

#ifdef CKPT

#include <src/pm/mpirun/mpirun_rsh.h>
#include "mpirun_dbg.h"
#include "common_ckpt.h"

#ifdef CR_FTB
#include <libftb.h>
#endif

// Initialize CR
// - it should be called only once at mpirun_rsh startup
// - it should *not* be called after restart
int CR_initialize();

// Finalize CR
// - it should be called only once at mpirun_rsh ending
// - it should *not* be called before restart
int CR_finalize();

// Start CR thread
// - it can be called again after CR_stop_thread() has been called
// - it should be called after restart
int CR_thread_start( unsigned int nspawns );

// Stop CR thread
// - it can be called again after CR_start_thread() has been called
// - it should be called when restarting
// If blocking is set, it will wait for the CR thread to terminate
int CR_thread_stop( int blocking );


char *create_mpispawn_vars(char *mpispawn_env);
void save_ckpt_vars_env(void);
void save_ckpt_vars(char *, char *);
void set_ckpt_nprocs(int nprocs);

#define CR_SESSION_MAX  16
extern char sessionid[CR_SESSION_MAX];


// =====================================================
// For Migration

#ifdef CR_FTB
#define HOSTFILE_LEN 256
extern int nsparehosts;
extern int sparehosts_on;
extern char sparehostfile[HOSTFILE_LEN + 1];
extern char **sparehosts;
extern struct spawn_info_s *spawninfo;

int read_sparehosts(char *hostfile, char ***hostarr, int *nhosts);
int cr_ftb_aggre_based_mig(char *src);
extern int cr_ftb_callback(FTB_receive_event_t *, void *);
#endif /* CR_FTB */

// =====================================================

#endif /* CKPT */ 


#endif
