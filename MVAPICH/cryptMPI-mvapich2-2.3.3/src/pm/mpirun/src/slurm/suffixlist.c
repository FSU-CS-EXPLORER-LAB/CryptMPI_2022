/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#include <suffixlist.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct suffix_range_iterator_t * sr_iterator;

typedef struct suffix_range_t {
    char const * begin;
    char const * end;
} suffix_range;

typedef struct suffix_range_iterator_t {
    suffix_range sr;
    char next[];
} suffix_range_iterator;

typedef struct suffix_node_t {
    sr_handle sr;
    struct suffix_node_t * next;
} suffix_node;

typedef struct suffix_list_t {
    suffix_node * last;
    suffix_node sentinel;
} suffix_list;

typedef struct suffix_list_iterator_t {
    suffix_list sl;
    sr_iterator sri;
    suffix_node * sn;
} suffix_list_iterator;

static int
increment_char (char * const c)
{
    switch (*c) {
        case '9':
            *c = '0';
            return 1;
        default:
            (*c)++;
            return 0;
    }
}

static void
increment_suffix (char * const suffix, size_t const len)
{
    char buffer[len + 2];
    size_t i = len;
    
    buffer[0] = '0';
    strcpy(buffer + 1, suffix);

    while (increment_char(&(buffer[i]))) i--;

    if (i) {
        strcpy(suffix, buffer + 1);
    }

    else {
        strcpy(suffix, buffer);
    }
}

static suffix_node *
sn_create (sr_handle sr)
{
    suffix_node * node = malloc(sizeof (suffix_node));

    if (node) {
        node->sr = sr;
        node->next = NULL;
    }

    return node;
}

static void
sn_destroy (suffix_node * node)
{
    sr_destroy(node->sr);
    free(node);
}

static sr_iterator
sri_create (sr_handle sr)
{
    sr_iterator sri = (sr_iterator)sr;

    sri->next[0] = '\0';

    return sri;
}

static char const *
sri_next (sr_iterator sri)
{
    if ('\0' != sri->next[0]) {
        increment_suffix(sri->next, strlen(sri->next));
    }

    else {
        strcpy(sri->next, sri->sr.begin);
    }

    return sri->next;
}

static int
sri_hasNext (sr_iterator sri)
{
    return strcmp(sri->next, sri->sr.end);
}

sr_handle
sr_create (char const * const begin, char const * const end)
{
    sr_handle sr = malloc(sizeof (suffix_range_iterator));

    if (sr) {
        sr->begin = begin;
        sr->end = end;
    }

    return sr;
}

void
sr_destroy (sr_handle sr)
{
    free(sr);
}

sl_handle
sl_create (sr_handle sr)
{
    sl_handle sl = malloc(sizeof (suffix_list_iterator));

    if (sl) {
        sl->sentinel.sr = NULL;
        sl->last = sl->sentinel.next = &(sl->sentinel);

        if (!sl_insert(sl, sr)) {
            sl_destroy(sl);
            sl = NULL;
        }
    }
    
    return sl;
}

sl_handle
sl_insert (sl_handle sl, sr_handle sr)
{
    suffix_node * node = sn_create(sr);
   
    if (!node) {
        return NULL;
    }

    node->next = sl->last->next;
    sl->last = sl->last->next = node;

    return sl;
}

void
sl_destroy (sl_handle sl)
{
    suffix_node * node, * next;

    for (node = sl->sentinel.next; node->sr; node = next) {
        next = node->next;
        sn_destroy(node);
    }

    free(sl);
}

sl_iterator
sli_create (sl_handle sl)
{
    sl_iterator sli = (sl_iterator)sl;

    sli->sn = sl->sentinel.next;
    sli->sri = sri_create(sli->sn->sr);

    return sli;
}

char const *
sli_next (sl_iterator sli)
{
    if (!sri_hasNext(sli->sri)) {
        sli->sn = sli->sn->next;
        sli->sri = sri_create(sli->sn->sr);
    }

    return sri_next(sli->sri);
}

int
sli_hasNext (sl_iterator sli)
{
    return sri_hasNext(sli->sri) || sli->sn->next->sr;
}
