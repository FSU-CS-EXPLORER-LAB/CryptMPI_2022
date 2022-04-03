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

#ifndef MPISPAWN_CKPT_H
#define MPISPAWN_CKPT_H

#include "mpichconf.h"

#ifdef CKPT

#include "common_ckpt.h"

#define CR_RESTART_CMD      "cr_restart"

// Used in mpispawn.c TODO: try remove these dependencies
extern char *sessionid;
extern char ckpt_filename[CR_MAX_FILENAME];
extern int restart_context;
extern int checkpoint_count;

int CR_Init(int);
void cr_cleanup(void);

#ifdef CR_FTB

extern volatile int cr_mig_spare_cond;
extern volatile int cr_mig_src_can_exit;
extern int eNCHILD;
extern volatile int cr_mig_src;
extern volatile int cr_mig_tgt;
extern volatile int num_migrations;
extern char cr_mig_src_host[32];
extern char cr_mig_tgt_host[32];

extern int *migrank;

struct spawn_info_s *spawninfo;
int exclude_spare;

extern FTB_client_handle_t ftb_handle;
extern FTB_event_info_t cr_ftb_events[];
extern pthread_t CR_wfe_tid;

void *CR_wait_for_errors(void *);
#else                           /* ! CR_FTB */

#endif                          /* CR_FTB */

extern int cr_spawn_degree;

#ifdef CR_AGGRE
extern int use_aggre_mig;       // whether we enable migration func in CRFS
extern int use_aggre;
int init_ckpt_aggregation();
#endif

int restart_mpi_process(int cached_cr_mig_tgt, int i);

#endif                          /* CKPT */

#endif                          /* MPISPAWN_CKPT_H */
