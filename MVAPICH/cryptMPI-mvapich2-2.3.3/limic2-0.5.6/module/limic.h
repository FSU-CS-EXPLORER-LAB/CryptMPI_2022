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
 * limic.h
 *  
 * LiMIC2:  Linux Kernel Module for High-Performance MPI Intra-Node 
 *          Communication
 * 
 * Author:  Hyun-Wook Jin <jinh@konkuk.ac.kr>
 *          System Software Laboratory
 *          Department of Computer Science and Engineering
 *          Konkuk University
 *
 * History: Jul 15 2007 Launch
 *
 *          Feb 27 2009 Modified by Karthik Gopalakrishnan (gopalakk@cse.ohio-state.edu)
 *                                  Jonathan Perkins       (perkinjo@cse.ohio-state.edu)
 *            - Automatically create /dev/limic
 *            - Add versioning to the Kernel Module
 *
 *          Oct 10 2009 Modified by Hyun-Wook Jin
 *            - Fragmented memory mapping & data copy
 *
 */

#ifndef _LIMIC_INCLUDED_
#define _LIMIC_INCLUDED_

#include <linux/init.h>
#include <linux/module.h> 
#include <linux/cdev.h>
#include <linux/types.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/version.h>
#include <linux/highmem.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <asm/page.h>
#include <asm/uaccess.h>
#include <linux/mm.h>
#include <linux/pagemap.h>
#include <asm/pgtable.h>
#include <linux/sched.h>
#ifdef HAVE_LIMIC_LOCK
#include <linux/mutex.h>
#endif

#define LIMIC_MODULE_MAJOR 0
#define LIMIC_MODULE_MINOR 7

/*
 * Account for changes in device_create and device_destroy
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,13)
#   define CREATE_LIMIC_DEVICE
#   if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,15)
#       define device_create(cls, parent, devt, device, ...) \
            class_device_create(cls, devt, device, __VA_ARGS__)
#   elif LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
#       define device_create(cls, parent, devt, device, ...) \
            class_device_create(cls, parent, devt, device, __VA_ARGS__)
#   elif LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26)
#       define device_create(cls, parent, devt, device, ...) \
            device_create(cls, parent, devt, __VA_ARGS__)
#   elif LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27)
#       define device_create            device_create_drvdata
#   endif
#   if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
#       define device_destroy   class_device_destroy
#   endif
#endif

/* /dev file name */
#define DEV_NAME "limic"
#define DEV_CLASS "limic"

#define LIMIC_TX      0x1c01
#define LIMIC_RX      0x1c02
#define LIMIC_VERSION 0x1c03
#define LIMIC_TXW     0x1c04

#define LIMIC_TX_DONE    1
#define LIMIC_RX_DONE    2
#define LIMIC_VERSION_OK 3
#define LIMIC_TXW_DONE   4

#define NR_PAGES_4_FRAG (16 * 1024)

#ifdef HAVE_LIMIC_LOCK
struct mutex limic_lock;
#endif

typedef struct limic_user {
    int nr_pages;   /* pages actually referenced */
    size_t offset;     /* offset to start of valid data */
    size_t length;     /* number of valid bytes of data */

    unsigned long va;
    void *mm;        /* struct mm_struct * */
    void *tsk;       /* struct task_struct * */
} limic_user;

typedef struct limic_request {
    void *buf;       /* user buffer */
    size_t len;         /* buffer length */
    limic_user *lu;  /* shandle or rhandle */
} limic_request;

typedef enum {
    CPY_TX,
    CPY_RX
} limic_copy_flag;


int limic_map_and_copy( limic_request *req, 
                        struct page **maplist, 
                        limic_copy_flag flag )
{
    limic_user *lu = req->lu;
    int pg_num = 0, ret = 0;
    size_t  offset = lu->offset;
    size_t pcount, len = (lu->length>req->len)?req->len:lu->length;
    void *kaddr, *buf = req->buf;
	
    lu->length = len; 
    while( ( pg_num <= lu->nr_pages ) && ( len > 0 ) ){
        pcount = PAGE_SIZE - offset;
        if (pcount > len)
            pcount = len;
	
        kaddr = kmap(maplist[pg_num]);

        if( flag == CPY_TX ){
            if( copy_from_user(kaddr+offset, buf, pcount) ){
                printk("LiMIC: (limic_map_and_copy) copy_from_user() is failed\n");
                return -EFAULT;
            }
        }
        else if( flag == CPY_RX ){
            if( copy_to_user(buf, kaddr+offset, pcount) ){
                printk("LiMIC: (limic_map_and_copy) copy_to_user() is failed\n");
                return -EFAULT;
            }
        }
	/* flush_dcache_page(maplist[pg_num]); */
        kunmap(maplist[pg_num]);

        len -= pcount;
        buf += pcount;
        ret += pcount;
        pg_num++;
        offset = 0;
    }

    return 0;
}


int limic_map_and_txcopy(limic_request *req, struct page **maplist)
{
    return limic_map_and_copy(req, maplist, CPY_TX);
}


int limic_map_and_rxcopy(limic_request *req, struct page **maplist)
{
    return limic_map_and_copy(req, maplist, CPY_RX);
}


void limic_release_pages(struct page **maplist, int pgcount)
{
    int i;
    struct page *map;
	
    for (i = 0; i < pgcount; i++) {
        map = maplist[i];
        if (map) {
            /* FIXME: cache flush missing for rw==READ
             * FIXME: call the correct reference counting function
             */
            page_cache_release(map); 
         }
    }

    kfree(maplist);
}


struct page **limic_get_pages(limic_user *lu, int rw)
{
    int err, pgcount;
    struct mm_struct *mm;
    struct page **maplist;

    mm = lu->mm;
    pgcount = lu->nr_pages;

    maplist = kmalloc(pgcount * sizeof(struct page **), GFP_KERNEL);
    if (unlikely(!maplist)) 
        return NULL;
	 
    /* Try to fault in all of the necessary pages */
    down_read(&mm->mmap_sem); 
    err = get_user_pages(lu->tsk, mm, lu->va, pgcount,
                         (rw==READ), 0, maplist, NULL); 
    up_read(&mm->mmap_sem);

    if (err < 0) { 
        limic_release_pages(maplist, pgcount); 
        return NULL;
    }
    lu->nr_pages = err;
 
    while (pgcount--) {
        /* FIXME: flush superflous for rw==READ,
         * probably wrong function for rw==WRITE
         */
        /* flush_dcache_page(maplist[pgcount]); */
    }
	
    return maplist;
}


int limic_get_info(void *buf, size_t len, limic_user *lu)
{
    limic_user limic_u;
    unsigned long va;
    int pgcount;

    va = (unsigned long)buf;
    limic_u.va = va;
    limic_u.mm = (void *)current->mm;
    limic_u.tsk = (void *)current;

    pgcount = (va + len + PAGE_SIZE - 1)/PAGE_SIZE - va/PAGE_SIZE; 
    if( !pgcount ){
        printk("LiMIC: (limic_get_info) number of pages is 0\n");
        return -EINVAL; 
    }       
    limic_u.nr_pages = pgcount;
    limic_u.offset = va & (PAGE_SIZE-1);
    limic_u.length = len;

    if( copy_to_user(lu, &limic_u, sizeof(limic_user)) ){
        printk("LiMIC: (limic_get_info) copy_to_user fail\n");
        return -EINVAL; 
    }       

    return 0;
}

#endif
