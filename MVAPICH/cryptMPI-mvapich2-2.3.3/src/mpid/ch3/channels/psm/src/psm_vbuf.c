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

/* vbuf.c taken from mrail and simplified for psm channel - tejus */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mpidimpl.h"

#include "psmpriv.h"
#include "psm_vbuf.h"
#include <sys/types.h>
#include <unistd.h>
#include <features.h>


/* head of list of allocated vbuf regions */
static vbuf_region *vbuf_region_head = NULL;
/*
 * free_vbuf_head is the head of the free list
 */
static vbuf *free_vbuf_head = NULL;

static int vbuf_n_allocated = 0;
static long num_free_vbuf = 0;
static long num_vbuf_get = 0;
static long num_vbuf_freed = 0;

static pthread_spinlock_t vbuf_lock;

int psm_init_vbuf_lock()
{
    int mpi_errno = MPI_SUCCESS;

    if (pthread_spin_init(&vbuf_lock, 0))
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno,
            MPIR_ERR_FATAL,
            "init_vbuf_lock",
            __LINE__,
            MPI_ERR_OTHER,
            "**fail",
            "%s: %s",
            "pthread_spin_init",
            strerror(errno));
    }
    return mpi_errno;
}

void free_1sc_req (struct vbuf* vbuf_head)
{
    struct vbuf* v = vbuf_head;
    MPID_Request * req;

    while (v) {
        if (v->req != NULL) {
            req = (MPID_Request*) v->req;
            /* Set the psm_flags to 0 so that we don't deadlock in a recursive call */
            req->psm_flags = 0;
            MPIDI_CH3_Request_destroy(req);
            v->req = NULL;
        }
        v = v->next;
    }

    return;
}

void psm_deallocate_vbuf()
{
    vbuf_region *r = vbuf_region_head;
    vbuf_region *next = NULL;

    pthread_spin_lock(&vbuf_lock);
    while (r)
    {
        next = r->next;
        MPIU_Memalign_Free(r->malloc_start);
        MPIU_Memalign_Free(r->malloc_buf_start);
        MPIU_Free(r);
        r = next;
    }
    pthread_spin_unlock(&vbuf_lock);
}

static int allocate_vbuf_region(int nvbufs)
{
    struct vbuf_region *reg = NULL;
    void *mem = NULL;
    void *vbuf_dma_buffer = NULL;
    int i = 0, alignment_vbuf = 64, alignment_dma = getpagesize();
    vbuf *cur = NULL;

    if (free_vbuf_head != NULL) {
        assert(0); 
    }

    reg = (struct vbuf_region *) MPIU_Malloc (sizeof(struct vbuf_region));
    if (NULL == reg) {
        goto mem_err;
    }

    if (posix_memalign((void**) &mem, alignment_vbuf, nvbufs * sizeof(vbuf))) {
        goto mem_err;
    }
   
    posix_memalign((void **)&vbuf_dma_buffer, alignment_dma, nvbufs * PSM_VBUFSZ);
    if (NULL == vbuf_dma_buffer) {
        goto mem_err;
    }
    
    memset(mem, 0, nvbufs * sizeof(vbuf));
    memset(vbuf_dma_buffer, 0, nvbufs * PSM_VBUFSZ);

    vbuf_n_allocated += nvbufs;
    num_free_vbuf += nvbufs;
    reg->malloc_start = mem;
    reg->malloc_buf_start = vbuf_dma_buffer;
    reg->malloc_end = (void *) ((char *) mem + nvbufs * sizeof(vbuf));
    reg->malloc_buf_end = (void *) ((char *) vbuf_dma_buffer + nvbufs * PSM_VBUFSZ);

    reg->count = nvbufs;
    free_vbuf_head = mem;
    reg->vbuf_head = free_vbuf_head;

    /* init the free list */
    for (i = 0; i < nvbufs - 1; ++i)
    {
        cur = free_vbuf_head + i;
        cur->next = free_vbuf_head + i + 1;
        cur->region = reg;
        cur->buffer = (unsigned char *) ((char *)vbuf_dma_buffer
            + i * PSM_VBUFSZ);
        cur->req = NULL;
    }

    /* last one needs to be set to NULL */
    cur = free_vbuf_head + nvbufs - 1;
    cur->region = reg;
    cur->next = NULL;
    cur->buffer = (unsigned char *) ((char *)vbuf_dma_buffer
        + (nvbufs - 1) * PSM_VBUFSZ);

    /* thread region list */
    reg->next = vbuf_region_head;
    vbuf_region_head = reg;

    return MPI_SUCCESS;

mem_err:
    return MPI_ERR_NO_MEM;
}

/* this function is only called by the init routines.
 * Cache the nic handle and ptag for later vbuf_region allocations.
 */
int psm_allocate_vbufs(int nvbufs)
{
    return allocate_vbuf_region(nvbufs);
}

vbuf* psm_get_vbuf()
{
    vbuf* v = NULL;

   	pthread_spin_lock(&vbuf_lock);

    /*
     * It will often be possible for higher layers to recover
     * when no vbuf is available, but waiting for more descriptors
     * to complete. For now, just abort.
     */
    if (NULL == free_vbuf_head) {
        allocate_vbuf_region(PSM_SECONDARY_POOL_SZ);

        if (NULL == free_vbuf_head) {
            goto mem_err;
        }
    }

    v = free_vbuf_head;
    v->req = NULL;
    --num_free_vbuf;
    ++num_vbuf_get;

    free_vbuf_head = free_vbuf_head->next;

    pthread_spin_unlock(&vbuf_lock);

    return(v);

mem_err:    
    return NULL;
}

void psm_release_vbuf(vbuf* v)
{
    MPID_Request * req;

    pthread_spin_lock(&vbuf_lock);

    MPIU_Assert(v != free_vbuf_head);
    v->next = free_vbuf_head;

    if (v->req != NULL) {
        req = (MPID_Request*) v->req;
        MPIU_Assert(MPIU_Object_get_ref(req) == 0);
        if(!(req->psm_flags & (PSM_1SIDED_PUTREQ | PSM_CONTROL_PKTREQ))) {
            MPIDI_CH3_Request_destroy(req);
        }
        v->req = NULL;
    }

    free_vbuf_head = v;
    ++num_free_vbuf;
    ++num_vbuf_freed;

    pthread_spin_unlock(&vbuf_lock);
}

/* vi:set sw=4 tw=80: */
