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

#ifndef CR_H
void MPIDI_CH3I_CR_lock (void);
void MPIDI_CH3I_CR_unlock (void);
void Wait_for_CR_Completion (void);

#define CR_H

#endif
