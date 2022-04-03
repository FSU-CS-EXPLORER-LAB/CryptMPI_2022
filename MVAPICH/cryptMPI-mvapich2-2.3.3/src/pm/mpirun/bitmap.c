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

#include <mpichconf.h>

#ifdef CR_AGGRE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "bitmap.h"

/*
Init a bitmap with "size" num of bits. All bits are set to "initval"(0 or 1)
*/
int bmp_init(struct bitmap *bmp, int size, int initval)
{
    if (size > MAX_BITMAP_SIZE) {
        printf("Error:  rqst bitmap size %d > max-size %d\n", size, MAX_BITMAP_SIZE);
        return -1;
    }

    pthread_mutex_init(&bmp->mutex, NULL);

    bmp->size = size;
    if (initval == 0) {
        memset((void *) bmp->bmp, 0, (MAX_BITMAP_SIZE) / 8);
    } else {
        memset((void *) bmp->bmp, 0xff, (MAX_BITMAP_SIZE) / 8);
    }

    return 0;
}

void bmp_destroy(struct bitmap *bmp)
{
    // bitmap is protected by upper layer lock, so this mutex is not needed
    pthread_mutex_destroy(&bmp->mutex);
}

/*
Find first bit "1" in the bitmap, toggle it to "0", and return its position in the map.
LSB's pos=0, with MSB's pos = size-1
Return -1 if no bit "1" exists 
*/
int bmp_ffs_and_toggle(struct bitmap *bmp)
{
    unsigned char *c;
    int max;
    unsigned int i, j;          // both i,j are bit-pos

    for (i = 0; i < bmp->size; i += 8) {
        c = ((unsigned char *) bmp->bmp) + i / 8;   // pos of the bit
        max = bmp->size - i;    // maximum bit pos in the unit c
        max = max < 8 ? max : 8;
        for (j = 0; j < max; j++) {
            if (*c & (unsigned char) (1 << j)) {    // find a bit "1"
                *c &= (unsigned char) (~(1 << j));  // toggle it to 0
                return (i + j);
            }
        }
    }
    printf("%s: Error: not bit 1 avail\n", __func__);
    //bmp_dump( bmp );  
    return -1;
}

/*
Find first bit "0" in the bitmap, toggle it to "1", and return its position in the map.
LSB's pos=0, with MSB's pos = size-1
Return -1 if no bit "1" exists 
*/
int bmp_ffz_and_toggle(struct bitmap *bmp)
{
    unsigned char *c;           // 
    int max;
    unsigned int i, j;          /// both i,j are bit-pos

    for (i = 0; i < bmp->size; i += 8) {
        c = (unsigned char *) bmp->bmp + i / 8; // pos of the bit
        max = bmp->size - i;    // maximum bit pos in the unit
        max = max <= 8 ? max : 8;
        for (j = 0; j < max; j++) {
            if ((*c & (unsigned char) (1 << j)) == 0) { // find a bit "0"
                *c |= (unsigned char) (1 << j); // toggle it to 1
                return (i + j);
            }
        }
    }
    printf("%s: Error: not bit 0 avail\n", __func__);
    return -1;
}

void bmp_dump(struct bitmap *bmp)
{
    printf("bitmap size = %d, bits are:\n\t", bmp->size);
    int i;
    for (i = 0; i < bmp->size; i++) {   //  i>=0; i-- ){
        printf("%c", bmp_get_pos(bmp, i) ? '1' : '0');
    }

    printf("\n");
}

void bmp_test()
{
    struct bitmap bmp;
    int i;

    bmp_init(&bmp, 64, 1);
    i = bmp_ffs_and_toggle(&bmp);

    bmp_dump(&bmp);

    bmp_set_bit(&bmp, 63);
    bmp_set_bit(&bmp, 1);
/*  bmp_clear_bit(&bmp, 1);
    bmp_clear_bit(&bmp, 21);
    bmp_clear_bit(&bmp, 0);
    bmp_clear_bit(&bmp, 4);
    bmp_clear_bit(&bmp, 63);    */
    bmp_dump(&bmp);

    i = bmp_ffz_and_toggle(&bmp);
    printf("ffz ret %d\n", i);
    bmp_dump(&bmp);

    i = bmp_ffs_and_toggle(&bmp);
    printf("ffs ret %d\n", i);
    bmp_dump(&bmp);

}

#endif
