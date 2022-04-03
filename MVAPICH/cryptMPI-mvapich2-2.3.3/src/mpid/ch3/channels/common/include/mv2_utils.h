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

/*
 *
 * mv2_utils.h
 *
 * Various utilities for MV2.
 */

#ifndef _MV2_UTILS_H
#define _MV2_UTILS_H
/****Function Declarations****/
int user_val_to_bytes(char* value, const char* param); //Takes care of 'K', 'M' and 'G' present in user parameters


#endif
