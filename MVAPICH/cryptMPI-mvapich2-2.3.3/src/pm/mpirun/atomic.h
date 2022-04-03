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

#ifndef _ATOMIC_H
#define _ATOMIC_H

#include "opa_primitives.h"

typedef OPA_int_t atomic_t;

#define ATOMIC_INIT(i)  OPA_INT_T_INITIALIZER(i)

static inline int atomic_read(atomic_t * ptr)
{
    return OPA_load_int(ptr);
}

static inline void atomic_set(atomic_t * ptr, int val)
{
    OPA_store_int(ptr, val);
}

static inline void atomic_add(int i, atomic_t * v)
{
    OPA_add_int(v, i);
}

static inline void atomic_sub(int i, atomic_t * v)
{
    OPA_add_int(v, -i);
}

/**
 * Atomically subtracts @i from @v and returns
 * true if the result is zero, or false for all
 * other cases.
 */
static inline int atomic_sub_and_test(int i, atomic_t * v)
{
    int tmp = OPA_fetch_and_add_int(v, -i);
    return (tmp - i == 0);
}

/**
 * Atomically increments @v by 1.
 */
static inline void atomic_inc(atomic_t * v)
{
    OPA_incr_int(v);
}

/**
 * Atomically increments @v by 1
 * and returns true if the result is zero, or false for all
 * other cases.
 */
static inline int atomic_inc_and_test(atomic_t * v)
{
    int tmp = OPA_fetch_and_incr_int(v);
    return (tmp + 1 == 0);
}

/**
 * Atomically decrements @v by 1.  Note that the guaranteed
 * useful range of an atomic_t is only 24 bits.
 */
static inline void atomic_dec(atomic_t * v)
{
    OPA_decr_int(v);
}

/**
 * Atomically decrements @v by 1 and
 * returns true if the result is 0, or false for all other
 * cases.
 */
static inline int atomic_dec_and_test(atomic_t * v)
{
    return (OPA_decr_and_test_int(v) == 0);
}

/**
 * Atomically adds @i to @v and returns true
 * if the result is negative, or false when
 * result is greater than or equal to zero.
 */
static inline int atomic_add_negative(int i, atomic_t * v)
{
    int tmp = OPA_fetch_and_add_int(v, i);
    return (tmp + i < 0);
}

#endif
