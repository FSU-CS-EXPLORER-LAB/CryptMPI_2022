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

#include <m_state.h>
#include <debug_utils.h>

#include <stdlib.h>
#include <pthread.h>

static int m_exit_code = EXIT_SUCCESS;
static M_STATE m_state = M_INITIALIZE;
static pthread_cond_t state_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t state_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline const char * 
M_STATE_STR(const M_STATE state)
{
    switch (state) {
        case M_INITIALIZE:
            return "M_INITIALIZE";
        case M_LAUNCH:
            return "M_LAUNCH";
        case M_RUN:
            return "M_RUN";
        case M_EXIT:
            return "M_EXIT";
        case M_RESTART:
            return "M_RESTART";
        default:
            return "UNKNOWN STATE";
    }
}

M_STATE
m_state_transition (const M_STATE_SET old_state, const M_STATE new_state)
{
    M_STATE state;
    pthread_mutex_lock(&state_mutex);

    if (old_state & m_state) {
        m_state = new_state;
        pthread_cond_broadcast(&state_cond);
        PRINT_DEBUG(0, "m_state_transition %s -> %s\n", M_STATE_STR(old_state),
                M_STATE_STR(new_state));

    }

    state = m_state;
    pthread_mutex_unlock(&state_mutex);

    return state;
}

void
m_state_exit (void)
{
    pthread_mutex_lock(&state_mutex);
    m_state = M_EXIT;
    pthread_cond_broadcast(&state_cond);
    pthread_mutex_unlock(&state_mutex);
}

void
m_state_fail (void)
{
    extern int m_exit_code;

    m_exit_code = EXIT_FAILURE;
    m_state_exit();
}

int
m_state_get_exit_code (void)
{
    return m_exit_code;
}

M_STATE
m_state_get (void)
{
    return m_state;
}

M_STATE
m_state_wait_while (const M_STATE_SET state) {
    M_STATE current_state;

    pthread_mutex_lock(&state_mutex);
    while (state & m_state) pthread_cond_wait(&state_cond, &state_mutex);
    current_state = m_state;
    pthread_mutex_unlock(&state_mutex);

    return current_state;
}

M_STATE
m_state_wait_until (const M_STATE_SET state)
{
    M_STATE current_state;

    pthread_mutex_lock(&state_mutex);
    while (!(state & m_state)) pthread_cond_wait(&state_cond, &state_mutex);
    current_state = m_state;
    pthread_mutex_unlock(&state_mutex);

    return current_state;
}

