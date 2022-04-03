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

#ifndef __BITMAP__
#define __BITMAP__

#define MAX_BITMAP_SIZE    (1024)   // maximum num of bits in the bitmap,
    // also the max num of entries of the resource monitored by this bitmap

typedef struct bitmap {
    //volatile unsigned char    bmp[MAX_BITMAP_SIZE/8]; // mem used for this bitmap
    unsigned char bmp[MAX_BITMAP_SIZE / 8]; // mem used for this bitmap
    int size;                   // num of bits in this bitmap

    pthread_mutex_t mutex;      //lock to protect the bitstring
    // bitmap is protected by upper layer lock, so this mutex is not needed

} bitmap_t;

int bmp_init(struct bitmap *bmp, int size, int initval);
void bmp_destroy(struct bitmap *bmp);

int bmp_ffs_and_toggle(struct bitmap *bmp);
int bmp_ffz_and_toggle(struct bitmap *bmp);

static inline void bmp_set_bit(struct bitmap *bmp, int pos)
{
    unsigned char *c = bmp->bmp + (pos / 8);
    int rem = pos % 8;

    *c |= (unsigned char) (1 << rem);
}

static inline void bmp_clear_bit(struct bitmap *bmp, int pos)
{
    unsigned char *c = bmp->bmp + (pos / 8);
    int rem = pos % 8;

    *c &= (~((unsigned char) (1 << rem)));
}

static inline int bmp_get_pos(struct bitmap *bmp, int pos)
{
    unsigned char *c = bmp->bmp + (pos / 8);

    return (*c & (1 << (pos % 8))) ? 1 : 0;
}

void bmp_dump(struct bitmap *bmp);
void bmp_test();

#endif                          // __BITMAP__
