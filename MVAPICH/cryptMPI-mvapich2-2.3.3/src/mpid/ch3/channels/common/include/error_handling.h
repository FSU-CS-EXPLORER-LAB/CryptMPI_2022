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

#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

// Configure an error signal handler 
// for SIGSEGV, SIGFPE, SIGILL, SIGABRT, SIGBUS
// - backtrace: if not 0, print a backtrace when an error occurs
// - return value: on success, 0 is returned
extern int setup_error_sighandler( int backtrace );


// Set the core dump size according to coresize parameter
// - coresize: "unlimited" or string representation of a positive integer
// - return value: on success, 0 is returned
extern int set_coresize_limit( const char* coresize );

// Print backtrace of the current thread
extern int print_backtrace();

// Enable/disable backtrace on error
extern int show_backtrace;

#endif
