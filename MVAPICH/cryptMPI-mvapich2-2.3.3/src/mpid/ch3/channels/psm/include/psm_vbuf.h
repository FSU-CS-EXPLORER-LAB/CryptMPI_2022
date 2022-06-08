/*
 * Copyright (C) 1999-2001 The Regents of the University of California
 * (through E.O. Lawrence Berkeley National Laboratory), subject to
 * approval by the U.S. Department of Energy.
 *
 * Use of this software is under license. The license agreement is included
 * in the file MVICH_LICENSE.TXT.
 *
 * Developed at Berkeley Lab as part of MVICH.
 *
 * Authors: Bill Saphir      <wcsaphir@lbl.gov>
 *          Michael Welcome  <mlwelcome@lbl.gov>
 */

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

/* vbuf.h taken from mrail device and simplified for PSM - tejus */

#ifndef _VBUF_H_
#define _VBUF_H_

 /* 16k inline data for puts/gets/accums. 1024 byte buffer for packet
    and dataloop content */

#define  PSM_VBUFSZ             16384
#define  PSM_INITIAL_POOL_SZ    256
#define  PSM_SECONDARY_POOL_SZ  64

#define MAX_PREPOST 512
#define RCV_PREPOST 128

typedef struct vbuf
{
    struct vbuf_region* region;
    unsigned char* buffer;
    void *req;
    void *next;
} vbuf;

/*
 * Vbufs are allocated in blocks and threaded on a single free list.
 *
 * These data structures record information on all the vbuf
 * regions that have been allocated.  They can be used for
 * error checking and to un-register and deallocate the regions
 * at program termination.
 *
 */
typedef struct vbuf_region
{
    void* malloc_start;         /* used to free region later  */
    void* malloc_end;           /* to bracket mem region      */
    void* malloc_buf_start;     /* used to free DMA region later */
    void* malloc_buf_end;       /* bracket DMA region */
    struct vbuf* vbuf_head;     /* first vbuf in region       */
    struct vbuf_region* next;   /* thread vbuf regions        */
    int count;                  /* number of vbufs in region  */
} vbuf_region;

typedef struct psm_prepost_list
{
    vbuf *prepost_array[RCV_PREPOST];
    struct psm_prepost_list *next;
} psm_prepost_list_t;

int     psm_init_vbuf_lock();
int     psm_allocate_vbufs(int nvbufs);
vbuf*   psm_get_vbuf();
void    psm_release_vbuf(vbuf* v);
void    psm_deallocate_vbuf();
void    psm_release_vbuf(vbuf* v);

#endif
