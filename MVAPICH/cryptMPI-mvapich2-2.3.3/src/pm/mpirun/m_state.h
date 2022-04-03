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

#ifndef M_STATE_H
#define M_STATE_H 1

typedef enum {
    M_INITIALIZE    = 0x01,
    M_LAUNCH        = 0x02,
    M_RUN           = 0x04,
    M_EXIT          = 0x08,
    M_RESTART       = 0x10,
} M_STATE;

typedef unsigned M_STATE_SET;

M_STATE m_state_get (void);
M_STATE m_state_transition (const M_STATE_SET, const M_STATE);
M_STATE m_state_wait_while (const M_STATE_SET);
M_STATE m_state_wait_until (const M_STATE_SET);

void m_state_exit (void);
void m_state_fail (void);
int m_state_get_exit_code (void);

#endif
