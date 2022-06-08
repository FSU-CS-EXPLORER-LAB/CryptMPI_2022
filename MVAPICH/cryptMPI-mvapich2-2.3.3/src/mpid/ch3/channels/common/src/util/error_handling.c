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

#include <unistd.h>
#include <stdlib.h>
#include <execinfo.h>
#include <sys/resource.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include "debug_utils.h"


// Basic principle
//
// From signal(7):
// Signal such as SIGSEGV and SIGFPE, generated as a consequence 
// of executing a specific machine-language instruction are thread directed, 
// as are signals targeted at a specific thread using pthread_kill(3).
//
// It means that the signal handler will be executed in the thread 
// that caused the error. So, we can explore its stack using backtrace(3).




#define MAX_DEPTH 100

// Print backtrace of the current thread
int print_backtrace()
{
    void *trace[MAX_DEPTH];
    unsigned int trace_size;
    char **trace_strings;

    // Get backtrace and symbols
    trace_size = backtrace(trace, MAX_DEPTH);
    trace_strings = backtrace_symbols(trace, trace_size);
    if ( trace_strings == NULL ) {
        PRINT_ERROR( "backtrace_symbols: error\n" );
        return -1;
    }

    // Print backtrace
    unsigned int i;
    for ( i = 0 ; i < trace_size ; ++i )
    {
        PRINT_ERROR( "%3i: %s\n", i, trace_strings[i] );
    }

    // Free trace_strings allocated by backtrace_symbols()
    free(trace_strings);

    return 0;
}

// Enable/disable backtrace on error
int show_backtrace = 0;

// Signal handler for errors
void error_sighandler(int sig, siginfo_t *info, void *secret) {
    // Always print error
    PRINT_ERROR( "Caught error: %s (signal %d)\n", sys_siglist[sig], sig );
    // Show backtrace if required
    if (show_backtrace) print_backtrace();
    // Raise the signal again with default handler
    raise( sig );
}

int setup_error_sighandler_helper( int signal ) {
    int rv;

    // Get the current signal handler
    struct sigaction old_sa;
    rv = sigaction(signal , NULL, &old_sa);
    if ( 0 != rv ) {
        PRINT_ERROR_ERRNO( "sigaction(): failed to read old signal handler for signal %d", errno, signal );
        return -1;
    }

    // Check for an existing signal handler (eg setup by the user)
    if ( old_sa.sa_handler != SIG_DFL && old_sa.sa_handler != SIG_IGN ) {
        // Do not overwrite a signal handler setup by the user
        // Silently return
        return -2;
    }

    // Setup the new handler
    struct sigaction sa;
    sigemptyset (&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_RESETHAND;   
    sa.sa_sigaction = error_sighandler;
    rv = sigaction(signal , &sa, NULL);
    if ( 0 != rv ) {
        PRINT_ERROR_ERRNO( "sigaction(): failed to setup a new signal handler for signal %d", errno, signal );
        return -3;
    }

    return 0;
}

// Configure the error signal handler
int setup_error_sighandler( int backtrace ) {
    // Enable backtrace?
    show_backtrace = backtrace;

    // Setup the handler for these signals
    setup_error_sighandler_helper( SIGILL );
    setup_error_sighandler_helper( SIGABRT );
    setup_error_sighandler_helper( SIGFPE );
    setup_error_sighandler_helper( SIGSEGV );
    setup_error_sighandler_helper( SIGBUS );
    // All return codes are ignored because
    // this is not required for MVAPICH2 to work properly

    return 0;
}

// Set the core dump size according to coresize parameter
int set_coresize_limit( const char* coresize )
{
    if ( coresize != NULL && strcmp( coresize, "default" ) != 0 ) {
        struct rlimit core_limit;
        int rv;
        // read current rlimit structure
        rv = getrlimit( RLIMIT_CORE, &core_limit );
        if ( rv != 0 ) {
            PRINT_ERROR_ERRNO( "getrlimit", errno );
            return -1;
        }
        // update the rlimit structure
        if ( strcmp( coresize, "unlimited") == 0 ) {
            core_limit.rlim_cur = RLIM_INFINITY;
        } else {
            core_limit.rlim_cur = atoi( coresize );
        }
        // apply new rlimit structure
        rv = setrlimit(RLIMIT_CORE,&core_limit);
        if ( rv != 0 )
        {
            PRINT_ERROR_ERRNO( "setrlimit", errno );
            return -1;
        }
    }
    return 0;
}

