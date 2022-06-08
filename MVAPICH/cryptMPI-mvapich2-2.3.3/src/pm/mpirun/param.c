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

#include <mpirun_util.h>

/*
 * Array of mpirun parameters that should be forwarded
 */
static char const *const parameters[] = {
    "MV2_CKPT_FILE",
    "MV2_CKPT_INTERVAL",
    "MV2_CKPT_MAX_SAVE_CKPTS",
    "MV2_CKPT_USE_AGGREGATION",
    "MV2_FASTSSH_THRESHOLD",
    "MV2_NPROCS_THRESHOLD",
    "MV2_MPIRUN_TIMEOUT",
    "MV2_MT_DEGREE",
    "MPIEXEC_TIMEOUT",
    "MV2_CKPT_AGGREGATION_BUFPOOL_SIZE",
    "MV2_CKPT_AGGREGATION_CHUNK_SIZE",
    "MV2_CKPT_MAX_CKPTS",
    "MV2_IGNORE_SYSTEM_CONFIG",
    "MV2_IGNORE_USER_CONFIG",
    "MV2_USER_CONFIG",
    "MV2_DEBUG_CORESIZE",
    "MV2_DEBUG_SHOW_BACKTRACE",
};

static size_t const num_parameters = sizeof(parameters) / sizeof(char const *const);

/*
 * str must be dynamically allocated
 */
extern char *append_mpirun_parameters(char *str)
{
    extern size_t const num_parameters;
    char const *value;
    size_t i;

    for (i = 0; i < num_parameters; i++) {
        if ((value = getenv(parameters[i]))) {
            char *key_value = mkstr(" %s=%s", parameters[i], value);

            str = append_str(str, key_value);
            free(key_value);
        }
    }

    return str;
}
