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

#ifndef IB_ERRORS_H
#define IB_ERRORS_H

#include "upmi.h"

#define NEM_IB_ERR(message, args...) {                          \
    MPIU_Internal_error_printf("[%s:%d] [%s:%d] ",              \
            me.hostname, me.rank,                               \
            __FILE__, __LINE__);                                \
    MPIU_Internal_error_printf(message, ##args);                \
    MPIU_Internal_error_printf("\n");                           \
}

#ifdef DEBUG_NEM_IB
#define NEM_IB_DBG(message, args...) {                          \
    MPL_msg_printf("[%s:%d] [%s:%d] ", me.hostname, me.rank,   \
            __FILE__, __LINE__);                                \
    MPL_msg_printf(message, ##args);                           \
    MPL_msg_printf("\n");                                      \
}
#else
#define NEM_IB_DBG(message, args...)
#endif


#undef DEBUG_PRINT

#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    MPL_error_printf("[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    MPL_error_printf(args);                                    \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif


#define ibv_va_error_abort(code, message, args...)  {           \
    int my_rank;                                                \
    UPMI_GET_RANK(&my_rank);                                     \
    fprintf(stderr, "[%d] Abort: ", my_rank);                   \
    fprintf(stderr, message, ##args);                           \
    fprintf(stderr, " at line %d in file %s\n", __LINE__,       \
            __FILE__);                                          \
    fflush (stderr);                                            \
    exit(code);                                                 \
}


/**
 * Terminate the program with a specified message and error code.
 */
void ib_internal_error_abort(int line, char *file, int code, char *message);

/**
 * Terminate the program with a specified message and error code.
 */
#define ibv_error_abort(code, message)  ib_internal_error_abort(__LINE__, __FILE__, code, message)


#define GEN_EXIT_ERR     -1     /* general error which forces us to abort */
#define GEN_ASSERT_ERR   -2     /* general assert error */
#define IBV_RETURN_ERR   -3     /* ibverbs funtion return error */
#define IBV_STATUS_ERR   -4     /* ibverbs funtion status error */

#endif
