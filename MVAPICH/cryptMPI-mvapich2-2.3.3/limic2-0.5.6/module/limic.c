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
 * limic.c
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
 *            - Test compatibility between Kernel Module & User Space Library
 *          
 *          Oct 10 2009 Modified by Hyun-Wook Jin
 *            - Fragmented memory mapping & data copy
 *
 */

#include "limic.h"

MODULE_AUTHOR("Hyun-Wook Jin <jinh@konkuk.ac.kr>");
MODULE_DESCRIPTION("LiMIC2: Linux Kernel Module for High-Performance MPI Intra-Node Communication");
MODULE_VERSION("0.5.6");
MODULE_LICENSE("Dual BSD/GPL"); /* BSD only */

#ifdef HAVE_UNLOCKED_IOCTL
#define LiMIC2_IOCTL_IGNORED_ARGS struct file * file
typedef long (* LiMIC2_IOCTL)(LiMIC2_IOCTL_IGNORED_ARGS, unsigned int,
        unsigned long);
#else
#define LiMIC2_IOCTL_IGNORED_ARGS struct inode * inode, struct file * file
typedef int (* LiMIC2_IOCTL)(LiMIC2_IOCTL_IGNORED_ARGS, unsigned int,
        unsigned long);
#endif

struct cdev *limic_cdev;
static dev_t limic_devnum;

#ifdef CREATE_LIMIC_DEVICE
struct class  *limic_class;
struct device *limic_device;
#endif /* defined(CREATE_LIMIC_DEVICE) */

static int limic_ioctl(LiMIC2_IOCTL_IGNORED_ARGS, unsigned int,
        void *);
#ifdef HAVE_UNLOCKED_IOCTL
static int limic_unlocked_ioctl(LiMIC2_IOCTL_IGNORED_ARGS, unsigned int,
        void *);
#endif

static int limic_open(struct inode *, struct file *);
static int limic_release(struct inode *, struct file *);

static struct file_operations limic_fops = {
#ifdef HAVE_UNLOCKED_IOCTL
    .unlocked_ioctl = (LiMIC2_IOCTL) limic_unlocked_ioctl,
#else
    .ioctl          = (LiMIC2_IOCTL) limic_ioctl,
#endif
    .open           = limic_open,
    .release        = limic_release
};

static int limic_ioctl(LiMIC2_IOCTL_IGNORED_ARGS,
        unsigned int op_code,
        void * arg)
{
    limic_request req, frag_req;
    int err;
    size_t len_left, len_copied; 
    struct page **maplist;
    limic_user *lu, frag_lu;
    limic_user temp_lu;
    uint32_t vinfo;
    int pgcount;

    switch (op_code) {
        case LIMIC_VERSION:
            if (copy_from_user((void *)&vinfo, arg, sizeof(uint32_t)))
                return -EFAULT;

            /*
             * For the Kernel Module & User Space Library to interoperate:
             * - Major version should be equal for both
             * - Library's Minor version should be <= Module's Minor version
             */
            if ( ( (vinfo >> 16)    == LIMIC_MODULE_MAJOR ) &&
                 ( (vinfo & 0xFFFF) == LIMIC_MODULE_MINOR )    )
            {
                return LIMIC_VERSION_OK;
            }
            else {
                return -EINVAL;
            }

        case LIMIC_TX:
            if(copy_from_user((void *)&req, arg, sizeof(limic_request)))
                return -EFAULT;

            if((err = limic_get_info(req.buf, req.len, req.lu))) return err;

            return LIMIC_TX_DONE;

        case LIMIC_RX:
            if(copy_from_user((void *)&req, arg, sizeof(limic_request)))
                return -EFAULT;

            if(copy_from_user((void *)&temp_lu, req.lu, sizeof(limic_user)))
                return -EFAULT;

            lu = &temp_lu;
            
            /* init for the first mapping fragment */
            if(((lu->va & (PAGE_SIZE-1)) < lu->offset) || req.len < lu->length ){
                frag_lu.va = lu->va + (lu->offset - (lu->va & (PAGE_SIZE-1)));
                frag_lu.offset = (lu->offset)%PAGE_SIZE; 
                pgcount = (frag_lu.va + req.len + PAGE_SIZE - 1)/PAGE_SIZE - frag_lu.va/PAGE_SIZE;
            } else {
                frag_lu.va = lu->va;
                frag_lu.offset = lu->offset;
                pgcount = lu->nr_pages;
            }
       
            frag_lu.mm = lu->mm;
            frag_lu.tsk = lu->tsk;
            frag_lu.nr_pages = (pgcount < NR_PAGES_4_FRAG) ? pgcount : NR_PAGES_4_FRAG;
            frag_lu.length = frag_lu.nr_pages * PAGE_SIZE - frag_lu.offset;
            if(frag_lu.length > lu->length)
                frag_lu.length = lu->length;
            if(frag_lu.length > req.len)
                frag_lu.length = req.len;
            frag_req.lu = &frag_lu;
            len_left = (req.len < lu->length) ? req.len : lu->length;
            len_copied = 0;

            while (len_left > 0) { 
                /* setup for the destination buffer of this fragment */
                frag_req.buf = req.buf + len_copied;
                frag_req.len = req.len - len_copied;

                maplist = limic_get_pages(&frag_lu, READ);
                if(!maplist) return -EINVAL;

                if((err = limic_map_and_rxcopy(&frag_req, maplist))) {
                    limic_release_pages(maplist, frag_lu.nr_pages);
                    return err;
                }

                limic_release_pages(maplist, frag_lu.nr_pages);

                /* setup for next mapping fragment */
                frag_lu.offset = 0;
                frag_lu.va += frag_lu.length;
                len_left -= frag_lu.length;
                len_copied += frag_lu.length;
                frag_lu.length = (len_left < NR_PAGES_4_FRAG * PAGE_SIZE) ? len_left : NR_PAGES_4_FRAG * PAGE_SIZE;
                frag_lu.nr_pages = (frag_lu.length + PAGE_SIZE - 1)/PAGE_SIZE;
            } /* end of while */

            if(put_user(len_copied, &req.lu->length))
                return -EFAULT;

            return LIMIC_RX_DONE;
        case LIMIC_TXW:
            if(copy_from_user((void *)&req, arg, sizeof(limic_request)))
                return -EFAULT;

            if(copy_from_user((void *)&temp_lu, req.lu, sizeof(limic_user)))
                return -EFAULT;

            lu = &temp_lu;

            /* init for the first mapping fragment */
            if(((lu->va & (PAGE_SIZE-1)) < lu->offset) || req.len < lu->length ){
                frag_lu.va = lu->va + (lu->offset - (lu->va & (PAGE_SIZE-1)));
                frag_lu.offset = (lu->offset)%PAGE_SIZE;
                pgcount = (frag_lu.va + req.len + PAGE_SIZE - 1)/PAGE_SIZE - frag_lu.va/PAGE_SIZE;
            } else {
                frag_lu.va = lu->va;
                frag_lu.offset = lu->offset;
                pgcount = lu->nr_pages;
            }

            frag_lu.mm = lu->mm;
            frag_lu.tsk = lu->tsk;
            frag_lu.nr_pages = (pgcount < NR_PAGES_4_FRAG) ? pgcount : NR_PAGES_4_FRAG;
            frag_lu.length = frag_lu.nr_pages * PAGE_SIZE - frag_lu.offset;
            if(frag_lu.length > lu->length)
                frag_lu.length = lu->length;
            if(frag_lu.length > req.len)
                frag_lu.length = req.len;
            frag_req.lu = &frag_lu;
            len_left = (req.len < lu->length) ? req.len : lu->length;
            len_copied = 0;

            while (len_left > 0){

                /* setup for the destination buffer of this fragment */
                frag_req.buf = req.buf + len_copied;
                frag_req.len = req.len - len_copied;

                maplist = limic_get_pages(&frag_lu, READ);
                if(!maplist) return -EINVAL;

                if((err = limic_map_and_txcopy(&frag_req, maplist))) {
                    limic_release_pages(maplist, frag_lu.nr_pages);
                    return err;
                }

                limic_release_pages(maplist, frag_lu.nr_pages);

                /* setup for next mapping fragment */
                frag_lu.offset = 0;
                frag_lu.va += frag_lu.length;
                len_left -= frag_lu.length;
                len_copied += frag_lu.length;
                frag_lu.length = (len_left < NR_PAGES_4_FRAG * PAGE_SIZE) ? len_left : NR_PAGES_4_FRAG * PAGE_SIZE;
                frag_lu.nr_pages = (frag_lu.length + PAGE_SIZE - 1)/PAGE_SIZE;

            } /* end of while */

            if(put_user(len_copied, &req.lu->length))
                return -EFAULT;

            return LIMIC_TXW_DONE;

#if 0
        case OCK_RESET:
            while(module_refcount(THIS_MODULE) )
		module_put(THIS_MODULE);

	    try_module_get(THIS_MODULE);  
	    return OCK_RESETTED;
#endif
        default:
            return -ENOTTY;
    }

    return -EFAULT;
}


static int limic_open(struct inode *inode, struct file *fp)
{
    try_module_get(THIS_MODULE); 
    return 0;
}

#ifdef HAVE_UNLOCKED_IOCTL
static int limic_unlocked_ioctl(LiMIC2_IOCTL_IGNORED_ARGS,
        unsigned int op_code,
        void * arg)
{
    int ret;
#ifdef HAVE_LIMIC_LOCK
    mutex_lock(&limic_lock);
#endif

    ret = limic_ioctl(file, op_code, arg);
#ifdef HAVE_LIMIC_LOCK
    mutex_unlock(&limic_lock);
#endif
    return ret;
}
#endif

static int limic_release(struct inode *inode, struct file *fp)
{
    module_put(THIS_MODULE); 
    return 0;
}


int limic_init(void)
{
    int err;

    err = alloc_chrdev_region(&limic_devnum, 0, 1, DEV_NAME);
    if( err < 0 ){
        printk ("LiMIC: can't get a major number\n");
        goto err_alloc_chrdev_region;
    }

    limic_cdev = cdev_alloc();
    limic_cdev->ops = &limic_fops;
    limic_cdev->owner = THIS_MODULE;
    err = cdev_add(limic_cdev, limic_devnum, 1);
    if ( err < 0 ) {
        printk ("LiMIC: can't register the device\n");
        goto err_cdev_add;
    }

#ifdef CREATE_LIMIC_DEVICE
    limic_class = class_create(THIS_MODULE, DEV_NAME);
    if (IS_ERR(limic_class)) {
        printk ("LiMIC: can't create the %s class\n", DEV_CLASS);
        err = PTR_ERR(limic_class);
        goto err_class_create;
    }

    limic_device = device_create(limic_class, NULL, limic_devnum, NULL,
            DEV_NAME);

    if (IS_ERR(limic_device)) {
        printk ("LiMIC: can't create /dev/%s\n", DEV_NAME);
        err = PTR_ERR(limic_device);
        goto err_device_create;
    }
#endif /* defined(CREATE_LIMIC_DEVICE) */
#ifdef HAVE_LIMIC_LOCK
    mutex_init(&limic_lock);
#endif

    printk("LiMIC: module is successfuly loaded.\n");
    printk("LiMIC: device major number: %d.\n", MAJOR(limic_devnum));
#ifndef CREATE_LIMIC_DEVICE
    printk("LiMIC: use 'mknod /dev/%s c %d 0' to create the device file.\n",
            DEV_NAME, MAJOR(limic_devnum));
#endif /* not defined(CREATE_LIMIC_DEVICE) */

    return 0;

#ifdef CREATE_LIMIC_DEVICE
err_device_create:
    class_destroy(limic_class);

err_class_create:
    cdev_del(limic_cdev);
#endif /* defined(CREATE_LIMIC_DEVICE) */

err_cdev_add:
    unregister_chrdev_region(limic_devnum, 1);

err_alloc_chrdev_region:
    return err;
}


void limic_exit(void)
{
#ifdef HAVE_LIMIC_LOCK
    mutex_destroy(&limic_lock);
#endif

#ifdef CREATE_LIMIC_DEVICE
    device_destroy(limic_class, limic_devnum);
    class_destroy(limic_class);
#endif /* defined(CREATE_LIMIC_DEVICE) */
    cdev_del(limic_cdev);
    unregister_chrdev_region(limic_devnum, 1);
    printk("LiMIC: module is cleaned up.\n");
}


module_init(limic_init);
module_exit(limic_exit);
