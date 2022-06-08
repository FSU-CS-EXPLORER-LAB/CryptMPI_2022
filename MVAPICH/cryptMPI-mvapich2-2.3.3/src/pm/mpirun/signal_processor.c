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

#include <signal_processor.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_t sp_tid;

static int is_joined = 1;
static int is_running = 0;

typedef void (*func_t)(int);

struct sp_params {
    sigset_t sigmask;
    func_t processor;
    int copied;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

static void
cleanup_sp_thread (void * arg)
{
    extern int is_running;
    is_running = 0;
}

static void
sp_thread (struct sp_params * params)
{
    sigset_t sigmask = params->sigmask;
    func_t processor = params->processor;
    int error, signal;

    /*
     * Signal the completion of copying the signal mask and function pointer
     * to signal processor so any pending resources can be reclaimed.  Also
     * set the is_running variable.
     */
    pthread_mutex_lock(&params->mutex);
    params->copied = 1;
    is_running = 1;
    pthread_cond_signal(&params->cond);
    pthread_mutex_unlock(&params->mutex);

    pthread_cleanup_push(&cleanup_sp_thread, NULL);
    for (;;) {
        if ((error = sigwait(&sigmask, &signal))) {
            PRINT_ERROR_ERRNO("sigwait", error);
            abort();
        }

        processor(signal);
    }
    pthread_cleanup_pop(0);
}

extern void
start_sp_thread (sigset_t sigmask, void
        (*processor)(int), int detach_thread)
{
    pthread_attr_t attr;
    int error;
    struct sp_params params = {
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
        .copied = 0
    };

    params.sigmask = sigmask;
    params.processor = processor;

    if ((error = pthread_attr_init(&attr))) {
        PRINT_ERROR_ERRNO("pthread_attr_init", error);
        abort();
    }

    /*
     * This code is shared between mpirun_rsh and mpispawn.  In mpirun_rsh
     * we need the signal thread to be joinable but mpispawn is not yet
     * ready to join with this thread.
     */
    if (detach_thread) {
        error = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (error) {
            PRINT_ERROR_ERRNO("pthread_attr_setdetachstate", error);
            abort();
        }
    }

    error = pthread_sigmask(SIG_SETMASK, &params.sigmask, NULL);
    if (error) {
        PRINT_ERROR_ERRNO("pthread_sigmask", error);
        abort();
    }

    error = pthread_create(&sp_tid, &attr, (void * (*)(void *))&sp_thread,
            (void *)&params);
    if (error) {
        PRINT_ERROR_ERRNO("pthread_create", error);
        abort();
    }

    /*
     * Do not exit function until the newly initialized thread has copied
     * over the signal mask and pointer to the signal_processor.
     */
    pthread_mutex_lock(&params.mutex);
    while (!params.copied) pthread_cond_wait(&params.cond, &params.mutex);
    pthread_mutex_unlock(&params.mutex);

    /*
     * Done with mutex and cond
     */
    pthread_mutex_destroy(&params.mutex);
    pthread_cond_destroy(&params.cond);

    /*
     * This is so that stop_signal_processor does not try to join a detached
     * thread.
     */
    if (!detach_thread) {
        is_joined = 0;
    }
}

extern void
stop_sp_thread (void)
{
    extern pthread_t sp_tid;
    extern int is_joined;
    extern int is_running;
    void * return_value;

    if (is_running) pthread_cancel(sp_tid);
    if (!is_joined) pthread_join(sp_tid, &return_value);
    is_joined = 1;
}

/*
 * Use this function to unset any blocked or ignored signals.  It is important
 * to do this for any forked processes that we want to respond to signals
 * normally.
 */
extern void
clear_sigmask (void)
{
    sigset_t sigmask;

    sigemptyset(&sigmask);
    sigprocmask(SIG_SETMASK, &sigmask, NULL);
}

/* vi:set sw=4 sts=4 tw=76 expandtab: */
