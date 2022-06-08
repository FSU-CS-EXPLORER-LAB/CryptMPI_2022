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

enum OSU_INTERFACE_TYPE {
    OSU_GEN2,
    OSU_PSM,
    OSU_NEMESIS_IB,
};

#ifdef CHANNEL_MRAIL_GEN2
#   define OSU_INTERFACE OSU_GEN2
#endif

#ifdef CHANNEL_PSM
#   define OSU_INTERFACE OSU_PSM
#endif

#ifdef CHANNEL_NEMESIS_IB
#   define OSU_INTERFACE OSU_NEMESIS_IB
#endif
