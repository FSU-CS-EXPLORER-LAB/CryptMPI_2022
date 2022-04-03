/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H 1

#include <pthread.h>
#include <signal.h>
#include <debug_utils.h>

/*
 * This function creates a new thread which calls sigwait on the given
 * sigset_t.  if sigwait returns successful the function pointed to by the 2nd
 * argument is called with the signal caught by sigwait as its argument.
 *
 * If there is an error creating the thread will terminate the process after
 * printing out a diagnostic.  The created thread will also terminate the
 * process if there is an error detaching itself or if an invalid signal is
 * found in the signal set.
 *
 * Error messages are made using the PRINT_ERROR & PRINT_ERROR_ERRNO macros so
 * `set_output_prefix' should be called before calling this function.
 */
extern void start_sp_thread (sigset_t, void (*)(int), int);
extern void stop_sp_thread (void);
extern void clear_sigmask (void);

#endif
