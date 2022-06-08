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
#include "fuse_params.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <fuse.h>
#include <libgen.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <signal.h>

#include <infiniband/verbs.h>

#include "log.h"
#include "debug.h"
//#include "ib_server.h"
//#include "ib_client.h"

#include "ib_comm.h"
#include "openhash.h"
#include "ckpt_file.h"
#include "ib_buf.h"

#include "crfs.h"

#define CKPT_MIG                // enable ckpt-migration-FS
#define BUILDIN_MOD    (1)      //
extern hash_table_t *g_ht_cfile;
extern struct ib_HCA hca;

extern struct thread_pool *iopool;  // only crfs-client needs to expose a aggre-io-pool to the FS itself

////////////////////

extern int crfs_mode;           // CRFS works in Write-Aggre mode, or Proc-Mig mode
extern int mig_role;            // If CRFS in Mig-mode, I'm a client(src) or srv(tgt)?

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////
static int init_mode;           // 0: only as Write-Aggre, 1: Mig-cli,  2: Mig-srv

//// for Proc-Migration::
static int mig_state = 0;       // is a migration going on?
static mig_info_t minfo;

#if BUILDIN_MOD
static int pipe_fd;
#endif

//////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////

// Report errors to logfile and give -errno to caller
static inline int crfs_error(char *str)
{
    int ret = -errno;
    //log_msg("    %s: %s\n", str, strerror(errno));
    error("    %s: %s\n", str, strerror(errno));
    return ret;
}

/**
All the paths I see are relative to the root of the mounted filesystem.  
In order to get to the underlying filesystem, need to concatenate: 
(mount point dir in underlying FS,  relative-path in the mounted-FS)
Mount-point dir is saved to CRFS_DATA->rootdir in main().
Whenever I need a path for something I'll call this to construct it.
**/
static inline void crfs_fullpath(char *fpath, const char *path)
{
    strcpy(fpath, CRFS_DATA->rootdir);
    strncat(fpath, path, MAX_PATH_LEN); // ridiculously long paths will break here

    log_msg("    crfs_fullpath:  rootdir = \"%s\", path = \"%s\", fpath = \"%s\"\n", CRFS_DATA->rootdir, path, fpath);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
// Prototypes for all these functions, and the C-style comments,
// come indirectly from /usr/include/fuse.h
//

/**
 * Initialize filesystem * The return value will passed in the private_data field of
 * fuse_context to all file operations and as a parameter to the
 * destroy() method.
 *
 * Introduced in version 2.3
 * Changed in version 2.6
 */
// Undocumented but extraordinarily useful fact:  the fuse_context is
// set up before this function is called, and
// fuse_get_context()->private_data returns the user_data passed to
// fuse_main().  Really seems like either it should be a third
// parameter coming in here, or else the fact should be documented
// (and this might as well return void, as it did in older versions of
// FUSE).
void *crfs_init(struct fuse_conn_info *conn)
{
    dbg("****  begin...\n");
    dbg("\tprivate_data is %p\n", CRFS_DATA);
    // crfs_data is the fuse_get_context()->private_data, is the 
    // last param passed to fuse_main(..., priv_data)
    char ch;
#ifdef CKPT_MIG
    memset(&minfo, 0, sizeof(minfo));
    if (crfs_init_imp(init_mode) != 0) {
        err("Fail to init crfs...\n");
        ch = '1';
    } else
        ch = '0';
    dbg("****  crfs_mode=%d, mig_role=%d\n", crfs_mode, mig_role);
#if BUILDIN_MOD
    write(pipe_fd, &ch, 1);
    close(pipe_fd);
#endif

#endif

    /// NOTE:: the returned value is stored at fuse_get_context()->private_data.
    ///   This is very important!!

    return CRFS_DATA;           // is:fuse_get_context()->private_data
}

/**
Clean up filesystem. Called at "fusermount -u xxx"
Called on filesystem exit.
**/
void crfs_destroy(void *userdata)
{
    dbg("\ncrfs_destroy(userdata=0x%08x)\n", userdata);
    /// here, the userdata is   fuse_get_context()->private_data

#ifdef CKPT_MIG
    crfs_destroy_imp(init_mode);
#endif

    dbg("%s: now exit...\n", __func__);

}

/** Open directory
 *
 * This method should check if the open operation is permitted for
 * this  directory
 *
 * Introduced in version 2.3
 */
int crfs_opendir(const char *path, struct fuse_file_info *fi)
{
    DIR *dp;
    int retstat = 0;
    char fpath[MAX_PATH_LEN];

    //log_msg("\ncrfs_opendir(path=\"%s\", fi=0x%08x)\n", path, fi);
    dbg("\ncrfs_opendir(path=\"%s\", fi=0x%08x)\n", path, fi);
    crfs_fullpath(fpath, path);

    dp = opendir(fpath);
    if (dp == NULL)
        retstat = crfs_error("crfs_opendir opendir");

    fi->fh = (intptr_t) dp;

    log_fi(fi);

    return retstat;
}

/** Read directory
 *
 * This supersedes the old getdir() interface.  New applications
 * should use this.
 *
 * The filesystem may choose between two modes of operation:
 *
 * 1) The readdir implementation ignores the offset parameter, and
 * passes zero to the filler function's offset.  The filler
 * function will not return '1' (unless an error happens), so the
 * whole directory is read in a single readdir operation.  This
 * works just like the old getdir() method.
 *
 * 2) The readdir implementation keeps track of the offsets of the
 * directory entries.  It uses the offset parameter and always
 * passes non-zero offset to the filler function.  When the buffer
 * is full (or an error happens) the filler function will return
 * '1'.
 *
 * Introduced in version 2.3
 */
int crfs_readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi)
{
    int retstat = 0;
    DIR *dp;
    struct dirent *de;

//    log_msg("\ncrfs_readdir(path=\"%s\", buf=0x%08x, filler=0x%08x, offset=%lld, fi=0x%08x)\n",
//      path, buf, filler, offset, fi);
    dbg("    (path=\"%s\", buf=0x%08x, filler=0x%08x, offset=%lld, fi=0x%08x)\n", path, buf, filler, offset, fi);
    // once again, no need for fullpath -- but note that I need to cast fi->fh
    dp = (DIR *) (uintptr_t) fi->fh;

    dbg("readdir %s: pid is %d, thread-id = 0x%x\n", path, getpid(), pthread_self());

    // Every directory contains at least two entries: . and ..  If my
    // first call to the system readdir() returns NULL I've got an
    // error; near as I can tell, that's the only condition under
    // which I can get an error from readdir()
    de = readdir(dp);
    if (de == 0)
        return -errno;

    // This will copy the entire directory into the buffer.  The loop exits
    // when either the system readdir() returns NULL, or filler()
    // returns something non-zero.  The first case just means I've
    // read the whole directory; the second means the buffer is full.
    do {
        log_msg("calling filler with name %s\n", de->d_name);
        if (filler(buf, de->d_name, NULL, 0) != 0)
            return -ENOMEM;
    }
    while ((de = readdir(dp)) != NULL);

    log_fi(fi);

    return retstat;
}

/** Create a directory */
int crfs_mkdir(const char *path, mode_t mode)
{
    int retstat = 0;
    char fpath[PATH_MAX];

    //log_msg("\ncrfs_mkdir(path=\"%s\", mode=0%3o)\n",  path, mode);
    dbg("\ncrfs_mkdir(path=\"%s\", mode=0%3o)\n", path, mode);
    crfs_fullpath(fpath, path);

    retstat = mkdir(fpath, mode);
    if (retstat < 0)
        retstat = crfs_error("crfs_mkdir mkdir");

    return retstat;
}

/** Remove a directory */
int crfs_rmdir(const char *path)
{
    int retstat = 0;
    char fpath[PATH_MAX];

///    log_msg("crfs_rmdir(path=\"%s\")\n", path);
    dbg("crfs_rmdir(path=\"%s\")\n", path);
    crfs_fullpath(fpath, path);

    retstat = rmdir(fpath);
    if (retstat < 0)
        retstat = crfs_error("crfs_rmdir rmdir");

    return retstat;
}

/** Release directory
 *
 * Introduced in version 2.3
 */
int crfs_releasedir(const char *path, struct fuse_file_info *fi)
{
    int retstat = 0;

    //log_msg("\ncrfs_releasedir(path=\"%s\", fi=0x%08x)\n", path, fi);
    dbg("\ncrfs_releasedir(path=\"%s\", fi=0x%08x)\n", path, fi);
    log_fi(fi);

    closedir((DIR *) (uintptr_t) fi->fh);

    return retstat;
}

/** Synchronize directory contents
 * If the datasync parameter is non-zero, then only the user data
 * should be flushed, not the meta data
 * Introduced in version 2.3
 */
// when exactly is this called?  when a user calls fsync and it
// happens to be a directory? ???
int crfs_fsyncdir(const char *path, int datasync, struct fuse_file_info *fi)
{
    int retstat = 0;

    //log_msg("\ncrfs_fsyncdir(path=\"%s\", datasync=%d, fi=0x%08x)\n", path, datasync, fi);
    dbg("\ncrfs_fsyncdir(path=\"%s\", datasync=%d, fi=0x%08x)\n", path, datasync, fi);
    log_fi(fi);

    return retstat;
}

/** Get file attributes.
 *
 * Similar to stat().  The 'st_dev' and 'st_blksize' fields are
 * ignored.  The 'st_ino' field is ignored except if the 'use_ino'
 * mount option is given.
 */
int crfs_getattr(const char *path, struct stat *statbuf)
{
    int retstat = 0;
    char fpath[PATH_MAX];

    //dbg("%s: (path=\"%s\", statbuf=0x%08x)\n", __func__, path, statbuf);
    crfs_fullpath(fpath, path);

    retstat = lstat(fpath, statbuf);
    dbg("\t(path=\"%s\", statbuf=0x%08x), ret=%d\n", path, statbuf, retstat);
    if (retstat != 0) {
        // NOTE:: this is necessary, retstat = -errno
        retstat = crfs_error("crfs_getattr lstat");
        dbg("\terrno=%d, Cannot stat file: %s\n", errno, fpath);
    }
#ifdef CKPT_MIG
    //if( crfs_mode == CRFS_SRV )
    if (crfs_mode == MODE_MIG) {    // at srv-side: proc-migration needs the file-size to be large enough
        // to restart a proc
        if (S_ISREG(statbuf->st_mode)) {
            dbg("%s is file, set large size...\n", path);
            statbuf->st_size = 0x0ffffffffffL;  // set file size to 1TB
        }
    }
#endif

    log_stat(statbuf);

    return retstat;
}

/**
 * Create and open a file *
 * If the file does not exist, first create it with the specified
 * mode, and then open it. *
 * If this method is not implemented or under Linux kernel
 * versions earlier than 2.6.15, the mknod() and open() methods
 * will be called instead. *
 * Introduced in version 2.5
 */
int crfs_create(const char *path, mode_t mode, struct fuse_file_info *fi)
{
    int retstat = 0;
    char fpath[PATH_MAX];
    int fd;

    //log_msg("\ncrfs_create(path=\"%s\", mode=0%03o, fi=0x%08x)\n", path, mode, fi);
    crfs_fullpath(fpath, path);

    fd = creat(fpath, mode);
    dbg("\ncrfs_create(path=\"%s\", mode=0%03o, fi=0x%08x), get fd=%d\n", path, mode, fi, fd);
    if (fd < 0) {
        retstat = crfs_error("crfs_create creat");
        error("fail with: %s: %s\n", fpath, strerror(errno));
        return -EIO;
    }
    /////////////////////////////////////
#ifdef CKPT_MIG
    // lookup the hash-table for this cfile;
    //parse_ckpt_fname( path, &ckptid, &procrank);
    //ckpt_file_t* cfile = hash_table_create_record(g_ht_cfile, ckptid, procrank);
    ckpt_file_t *cfile = hash_table_create_record(g_ht_cfile, path);
    cfile->fd = fd;

    dbg("open ckpt file %s: ckptid %d, procrank %d, ref=%d, fd=%d\n", path, ckptid, procrank, cfile->ref, fd);
    fi->fh = (unsigned long) cfile;
    //dbg(" get cfile %p,fi->fh=0x%lx\n", cfile, fi->fh );

    ////////////////////////////////////
#else
    fi->fh = fd;
    log_fi(fi);
#endif

    return retstat;
}

/** File open operation
 *
 * No creation, or truncation flags (O_CREAT, O_EXCL, O_TRUNC)
 * will be passed to open().  Open should check if the operation
 * is permitted for the given flags.  Optionally open may also
 * return an arbitrary filehandle in the fuse_file_info structure,
 * which will be passed to all file operations.
 *
 * Changed in version 2.2
 */
int crfs_open(const char *path, struct fuse_file_info *fi)
{
    int fd;
    char fpath[PATH_MAX];

//    log_msg("\ncrfs_open(path\"%s\", fi=0x%08x)\n", path, fi);
    dbg("\ncrfs_open(path\"%s\", fi=0x%08x), flags=0%o\n", path, fi, fi->flags);
    crfs_fullpath(fpath, path);
    //fi->flags |= O_DIRECT;
    dbg("\ncrfs_open(path\"%s\", fi=0x%08x), flags=0%o\n", path, fi, fi->flags);

    fd = open(fpath, fi->flags);
    if (fd < 0) {
        crfs_error("crfs_open open");
        error("fail with: %s: %s\n", fpath, strerror(errno));
    }
    /////////////////////////////////////
#ifdef CKPT_MIG
    //lookup the hash-table for this cfile;

    //parse_ckpt_fname( path, &ckptid, &procrank);
    //ckpt_file_t* cfile = hash_table_create_record( g_ht_cfile, ckptid, procrank );

    ckpt_file_t *cfile = hash_table_create_record(g_ht_cfile, path);
    dbg("open ckpt file %s: ckptid %d, procrank %d, ref=%d, fd=%d\n", path, ckptid, procrank, cfile->ref, fd);

    cfile->fd = fd;             // record the file-handle       
    fi->fh = (unsigned long) cfile;

    ////////////////////////////////////
#else
    fi->fh = fd;
    log_fi(fi);
#endif

    return 0;                   //fd;
}

/** Read data from an open file
 *
 * Read should return exactly the number of bytes requested except
 * on EOF or error, otherwise the rest of the data will be
 * substituted with zeroes.  An exception to this is when the
 * 'direct_io' mount option is specified, in which case the return
 * value of the read system call will reflect the return value of
 * this operation.
 *
 * Changed in version 2.2
 */
// I don't fully understand the documentation above -- it doesn't
// match the documentation for the read() system call which says it
// can return with anything up to the amount of data requested. nor
// with the fusexmp code which returns the amount of data also
// returned by read.
int crfs_read(const char *path, char *buf, size_t size, off_t in_offset, struct fuse_file_info *fi)
{
    /***********************
    Attention:  fuse uses multi-thread internally to perform read. So, it's possible that multi-threads
    come here at same time, not in order of "offset".    
    Can disable this multi-thread behavior by: "./crfs xx yy  -s"
    ***********************/
    int retstat = 0;

    //log_msg("\ncrfs_read(path=\"%s\", buf=0x%08x, size=%d, offset=%lld, fi=0x%08x, ret=%d)\n",
    //  path, buf, size, in_offset, fi, retstat );

    /////////////////////////////////////////////////////
    // no need to get fpath on this one, since I work from fi->fh not the path
#ifdef CKPT_MIG
    ckpt_file_t *cfile = (ckpt_file_t *) fi->fh;
    if (!cfile) {
        error("Error: ckptfile %s: cfile is NULL...\n", path);
        return 0;
    }

    /*
       if((cnt++) % 2 == 0 ){
       printf("%s: First time, pause file %s...\n", __func__, path);
       sleep(3);
       }    */

    //if( crfs_mode == CRFS_CLIENT ) /// at ckpt-fs(client-side): normal read
    if (crfs_mode == MODE_WRITEAGGRE)   /// at ckpt-fs(client-side): normal read
    {
        retstat = pread(cfile->fd, buf, size, in_offset);
        if (retstat < 0)
            error(" pread ret %d: %s\n", retstat, strerror(errno));

        return retstat;
    }
    /// NOW, we are at mig-fs(srv-side), read partial-in-mem data
    off_t offset = in_offset;
    //dbg(" ===== Want cfile(%d,%d) @ (%d, %d):: ====== \n", cfile->ckpt_id, cfile->proc_rank, offset, size);   
    dbg("--- want cfile(%s) %d@%d --- \n", cfile->filename, size, in_offset);
    //dump_ckpt_file( cfile );

    ckpt_chunk_t *chunk;
    void *destbuf = buf;
    int to_read = size;
    int cpsize;

    while (to_read > 0) {
        chunk = cfile->curr_chunk;

        if (chunk) {
            if (offset < chunk->offset + chunk->curr_pos) { // this should never happen
                error("read offset %ld < chunk->offset %ld + curr_pos %ld\n", offset, chunk->offset, chunk->curr_pos);
                /// wait for next chunk?
                ckpt_free_chunk(chunk);
                cfile->curr_chunk = NULL;
                continue;
                //return 0;
            }
            if (offset >= (chunk->offset + chunk->size)) {  // this should never happen...
                error("read offset %ld > curr_chunk (%ld+%ld)\n", offset, chunk->offset, chunk->size);
                /// wait for next chunk?
                ckpt_free_chunk(chunk);
                cfile->curr_chunk = NULL;
                continue;       //return 0;
            }
            ///////
            //dbg("before copy:  to copy (%d, %d), chunk is::  ", offset, to_read);
            //dump_chunk( chunk );
            ////////////
            cpsize = chunk->size - chunk->curr_pos; // has this much avail data in this chunk
            cpsize = cpsize < to_read ? cpsize : to_read;

            ///dbg("copy src[400] = %d\n", *((char*)(chunk->buf+chunk->curr_pos+400)) );
            memcpy(destbuf, chunk->buf + chunk->curr_pos, cpsize);
            destbuf += cpsize;
            chunk->curr_pos += cpsize;
            to_read -= cpsize;
            offset += cpsize;
            ////////
            //dbg("After copy:  to copy (%d, %d), chunk is::  ", offset, to_read);
            //dump_chunk( chunk );
            //////////////////////

            // if this chunk has been completed, free it
            if (chunk->curr_pos >= chunk->size) {
                cfile->rcv_ready_chunk_num--;
                ckpt_free_chunk(chunk);
                free(chunk);    ////////
                cfile->curr_chunk = NULL;
                continue;
            }
        } else {                // the curr_chunk is NULL, need to get a chunk
            //dbg("wait for chunk at offset %d\n", offset);
            cfile->curr_chunk = get_chunk_from_ckpt_file(cfile, offset);
            if (cfile->curr_chunk == NULL)  // the ckpt-file is completed, ret
                break;
            if (cfile->curr_chunk->size == 0) { // a dummy chunk: signify the end of file               
                free(cfile->curr_chunk);
                cfile->curr_chunk = NULL;
                break;
            }
            //dbg("Get chunk at offset %d\n", offset);
        }
    }                           // end of while( to_read > 0 )

    /// has finished reading (size) data from ckpt
    dbg("====  Has read cfile(%s): %d@%d\n", cfile->filename, size - to_read, in_offset);
    //cfile->ckpt_id, cfile->proc_rank, size-to_read );
    //dump_ckpt_file( cfile );
    return (size - to_read);    //size;

#else
    retstat = pread(fi->fh, buf, size, in_offset);
    return retstat;
#endif

}

/** Write data to an open file
 *
 * Write should return exactly the number of bytes requested
 * except on error.  An exception to this is when the 'direct_io'
 * mount option is specified (see read operation).
 *
 * Changed in version 2.2
 */
int crfs_write(const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info *fi)
{
    int retstat = 0;

    // log_msg("\ncrfs_write(path=\"%s\", buf=0x%08x, size=%d, offset=%lld, fi=0x%08x)\n",
    //      path, buf, size, offset, fi );
    // no need to get fpath on this one, since I work from fi->fh not the path
    //log_fi(fi);

    //printf("%s: crfs_write(%s): %d@%d\n", __func__, path, size, offset );

    /////////////////////////////////////////////////
#ifdef CKPT_MIG
    ckpt_file_t *cfile = (ckpt_file_t *) fi->fh;
    if (!cfile) {
        error("Error: ckptfile %s: cfile is NULL...\n", path);
        return -EIO;
    }
    //if( crfs_mode == CRFS_SRV ) /// at mig-fs(srv-side): normal write
    if (crfs_mode == MODE_MIG)  /// at mig-fs(srv-side): normal write
    {
        retstat = pwrite(cfile->fd, buf, size, offset);
        if (retstat < 0)
            error("write ret %d: %s\n", retstat, strerror(errno));

        return retstat;
    }
    ///// NOW, we are at ckpt-fs(client-side): need write-aggre
    ckpt_chunk_t *chunk;
    const char *srcbuf = buf;
    int to_write = size;
    int cpsize;

    while (to_write > 0) {
        chunk = cfile->curr_chunk;

        if (chunk) {
            if (offset < chunk->offset + chunk->curr_pos) { // this should never happen
                error("write offset %ld < chunk->offset %ld + curr_pos %ld\n", offset, chunk->offset, chunk->curr_pos);
                /// wait for next chunk?
                ckpt_free_chunk(chunk);
                cfile->curr_chunk = NULL;
                continue;
                //return 0;
            }
            if (offset >= (chunk->offset + chunk->size)) {  // this should never happen...
                error("write offset %ld > curr_chunk (%ld+%ld)\n", offset, chunk->offset, chunk->size);
                /// wait for next chunk?
                ckpt_free_chunk(chunk);
                cfile->curr_chunk = NULL;
                continue;       //return 0;
            }
            ///////
            //dbg("before copy:  to copy (%d@%d), chunk is::  \n", to_write, offset);
            //dump_chunk( chunk );
            ////////////
            cpsize = chunk->size - chunk->curr_pos; // has this much avail space in this chunk
            cpsize = cpsize < to_write ? cpsize : to_write;

            memcpy(chunk->buf + chunk->curr_pos, srcbuf, cpsize);
            //dbg("after cp: tgt[400]=%d, src[400] = %d\n", 
            //*((char*)(chunk->buf+chunk->curr_pos+400)), srcbuf[400] );
            srcbuf += cpsize;
            chunk->curr_pos += cpsize;
            to_write -= cpsize;
            offset += cpsize;
            ////////
            //dbg("After copy:  to copy (%d@%d), chunk is::  ", to_write, offset);
            //dump_chunk( chunk );
            //// check the contents of the outgoing chunk...            
            //check_chunk_content( cfile, chunk, chunk->curr_pos );
            //////////////////////

            // has aggregated a full-chunk, pass it to an io-thr to write it
            if (chunk->curr_pos >= chunk->size) {
                dbg("*** (%s): pass lbuf:%d (%d@%d) to iothr...\n", path, chunk->bufid, chunk->curr_pos, chunk->offset);
                cfile->adv_size += chunk->curr_pos;
                atomic_inc(&minfo.chunk_cnt);
                workqueue_enqueue(iopool->queue, chunk, sizeof(*chunk), 0, 0);
                /////pwrite(cfile->fd, chunk->buf, chunk->curr_pos, chunk->offset);
                free(chunk);
                cfile->curr_chunk = NULL;
                continue;
            }
        } else {                // the curr_chunk is NULL, need to get a chunk
            //dbg("wait for chunk at offset %d\n", offset);
            chunk = alloc_ckpt_chunk();
            if (chunk == NULL) {
                err("fail to alloc a chunk...\n");
                break;
            }
            // may block here waiting for a free-buf-chunk
            //dbg("--- before: get free buf slot\n");
            chunk->bufid = get_buf_slot(hca.rdma_buf, &(chunk->buf), 0);
            //dbg("--- after:  get free-buf-slot = %d\n", chunk->bufid );

            chunk->ckpt_file = cfile;

            chunk->curr_pos = 0;    // r/w to this position
            chunk->size = hca.rdma_buf->slot_size;  // size of buf
            chunk->offset = offset; // this chunk is at this logical-offset of the whole file
            cfile->curr_chunk = chunk;

        }
    }                           // end of while( to_read > 0 )

    /// has finished reading (size) data from ckpt
    //dbg("====  Has write cfile %s: (%d,%d)  bytes:  %d\n", cfile->filename, 
    //  cfile->ckpt_id, cfile->proc_rank, size-to_write );
    //dump_ckpt_file( cfile );
    return (size - to_write);   //size;

    //////////////
#else
    retstat = pwrite(fi->fh, buf, size, offset);
    if (retstat < 0)
        retstat = crfs_error("crfs_write pwrite");
    return retstat;
#endif

}

/** Synchronize file contents. called at fsync(fd).
 *
 * If the datasync parameter is non-zero, then only the user data
 * should be flushed, not the meta data.
 *
 * Changed in version 2.2
 */
int crfs_fsync(const char *path, int datasync, struct fuse_file_info *fi)
{
//    log_msg("\ncrfs_fsync(path=\"%s\", datasync=%d, fi=0x%08x)\n", path, datasync, fi);
    //log_fi(fi);
    dbg("\ncrfs_fsync(path=\"%s\", datasync=%d, fi=0x%08x)\n", path, datasync, fi);

    /////////////////////////////////////////////////
#ifdef CKPT_MIG
    ckpt_file_t *cfile = (ckpt_file_t *) fi->fh;
    if (!cfile) {
        error("Error: ckptfile %s: cfile is NULL...\n", path);
        return -EIO;
    }
    /// at this point, app thinks this file is completed    
    gettimeofday(&cfile->tend, NULL);
//  unsigned long us = (cfile->tend.tv_sec - cfile->tstart.tv_sec)*1000000 + 
//                  (cfile->tend.tv_usec - cfile->tstart.tv_usec);
//  dbg("*** finished (%s), adv-size=%ld; cost %ld ms\n", path, cfile->adv_size, us/1000 );

    //if( crfs_mode == CRFS_CLIENT ) /// at ckpt-fs(client-side): flush the "curr_chunk"
    if (crfs_mode == MODE_WRITEAGGRE)   /// at ckpt-fs(client-side): flush the "curr_chunk"
    {
        ckpt_chunk_t *chunk = cfile->curr_chunk;

        if (chunk) {
            dbg("\t%s: flush final chunk\n", cfile->filename);
            chunk->is_last_chunk = 1;
            cfile->adv_size += chunk->curr_pos;
            atomic_inc(&minfo.chunk_cnt);
            workqueue_enqueue(iopool->queue, chunk, sizeof(*chunk), 0, 0);
            //////retstat = pwrite(cfile->fd, chunk->buf, chunk->curr_pos, chunk->offset);
            free(chunk);
            cfile->curr_chunk = NULL;
        }
        //else if(mig_role==ROLE_MIG_SRC )
        else {
            // send a dummy chunk (size=0) to mig-target:
            ckpt_chunk_t ck;    // a dummy chunk
            memset(&ck, 0, sizeof(ck)); // ck.size ==0, .curr_pos==0
            ck.ckpt_file = cfile;
            ck.is_last_chunk = 1;
            ck.bufid = -1;
            ck.curr_pos = 0;
            ck.offset = cfile->adv_size;
            atomic_inc(&minfo.chunk_cnt);
            dbg("***  fsync_(%s) %d@%d : send a dummy chunk as last_chk...\n", path, ck.curr_pos, ck.offset);
            workqueue_enqueue(iopool->queue, &ck, sizeof(ck), 0, 0);
            cfile->curr_chunk = NULL;
        }
    }
    return 0;

#else

    if (datasync)
        retstat = fdatasync(fi->fh);
    else
        retstat = fsync(fi->fh);

    if (retstat < 0)
        crfs_error("crfs_fsync fsync");

    return retstat;

#endif

}

/** Possibly flush cached data
 *
 * BIG NOTE: This is not equivalent to fsync().  It's not a
 * request to sync dirty data.
 NOTE:: this is invoked at app calling "close(fd)".
 *
 * Flush is called on each close() of a file descriptor.  So if a
 * filesystem wants to return write errors in close() and the file
 * has cached dirty data, this is a good place to write back data
 * and return any errors.  Since many applications ignore close()
 * errors this is not always useful.
 *
 * NOTE: The flush() method may be called more than once for each
 * open().  This happens if more than one file descriptor refers
 * to an opened file due to dup(), dup2() or fork() calls.  It is
 * not possible to determine if a flush is final, so each flush
 * should be treated equally.  Multiple write-flush sequences are
 * relatively rare, so this shouldn't be a problem.
 *
 * Filesystems shouldn't assume that flush will always be called
 * after some writes, or that if will be called at all.
 *
 * Changed in version 2.2
 */
int crfs_flush(const char *path, struct fuse_file_info *fi)
{
    int retstat = 0;
    /// the path is fullpath name + filename, relative to the mnt point

    //log_msg("\ncrfs_flush(path=\"%s\", fi=0x%08x)\n", path, fi);
    printf("\ncrfs_flush(path=\"%s\", fi=0x%p)\n", path, fi);
    // no need to get fpath on this one, since I work from fi->fh not the path

    ///////////////////////////////////////////////////////////
    // When app calls close(fd), this "flush" is invoked.
#ifdef CKPT_MIG
    ckpt_file_t *cfile = (ckpt_file_t *) fi->fh;
    if (!cfile)
        return retstat;

    char filename[128];
    strncpy(filename, path, 128);
    filename[128 - 1] = 0;
    hash_bucket_t *bkt = htbl_find_lock_bucket(g_ht_cfile, filename, strlen(filename));
    if (!bkt) {                 // the target file has already been cleaned from hash-tbl
        dbg(" other thr has del cfile-record for (%s)\n", filename);
        fi->fh = 0;
        return retstat;
    }
    //if( crfs_mode == CRFS_CLIENT ) /// at ckpt-fs(client-side): 
    if (crfs_mode == MODE_WRITEAGGRE)   // at ckpt-fs(client-side)
    {
        ///// 1.  flush the "curr_chunk" if any
        ckpt_chunk_t *chunk = cfile->curr_chunk;
        if (chunk) {
            dbg("\t%s: flush final chunk\n", cfile->filename);
            chunk->is_last_chunk = 1;
            cfile->adv_size += chunk->curr_pos;
            atomic_inc(&minfo.chunk_cnt);
            workqueue_enqueue(iopool->queue, chunk, sizeof(*chunk), 0, 0);
            free(chunk);
            cfile->curr_chunk = NULL;
        }
        /////  2. rm the record from hash-tbl       
        //pthread_mutex_lock( &cfile->mutex );
        cfile->can_release = 1;
        //pthread_mutex_unlock( &cfile->mutex );

        /// after setting "can_release", 
        /// an iothr may have cleaned the hash-tbl to remove this file
        //hash_bucket_t* bkt = htbl_find_lock_bucket(g_ht_cfile, filename, strlen(filename));
        //if( bkt )
        {                       // has located the cfile-record, and locked the hosting hash-bucket
            if (hash_table_put_record(g_ht_cfile, cfile, 1) > 0) {
                fi->fh = 0;
            }
        }
    }                           // end of if( CLIENT ... )

    htbl_unlock_bucket(bkt);

    return 0;
#endif
    ///////////////////////////////////////////////////////////

    return retstat;
}

/** Release an open file
 *
 * Release is called when there are no more references to an open
 * file: all file descriptors are closed and all memory mappings
 * are unmapped.
 *
 * For every open() call there will be exactly one release() call
 * with the same flags and file descriptor.  It is possible to
 * have a file opened more than once, in which case only the last
 * release will mean, that no more reads/writes will happen on the
 * file.  The return value of release is ignored.
 *
 * Changed in version 2.2
 */
int crfs_release(const char *path, struct fuse_file_info *fi)
{
    int retstat = 0;
    //log_msg("\ncrfs_release(path=\"%s\", fi=0x%08x)\n", path, fi);
    //log_fi(fi);
    printf("\ncrfs_release(path=\"%s\", fi=0x%p)\n", path, fi);

    ///////////////////////////////////////////////////////////
#ifdef CKPT_MIG
    ckpt_file_t *cfile = (ckpt_file_t *) fi->fh;
    if (!cfile)
        return retstat;

    hash_bucket_t *bkt = htbl_find_lock_bucket(g_ht_cfile, path, strlen(path));
    if (!bkt) {
        dbg(" other thr has del cfile-record for (%s)\n", path);
        fi->fh = 0;
        return retstat;
    }
    // now, has hold a lock on bucket
    ckpt_chunk_t *ck, *tmpck;
    dbg("now release ckptfile %s: \n", cfile->filename);

    //if( crfs_mode == CRFS_SRV ) // at mig-FS serv side: for migration
    if (crfs_mode == MODE_MIG)  // at mig-FS serv side: for migration
    {
        //pthread_mutex_lock( &cfile->mutex );
        if (!mv2_list_empty(&cfile->chunk_list)) {  /// clear all remaining chunk-bufs
            error("free cfile: ckpt-%d-proc-%d: chunk-list not empty!!!\n", cfile->ckpt_id, cfile->proc_rank);
            dump_ckpt_file(cfile);
            mv2_list_for_each_entry_safe(ck, tmpck, &(cfile->chunk_list), list) {
                //// free the buf-chunk
                free_buf_slot(hca.rdma_buf, ck->bufid, 0);
                mv2_list_del(&ck->list);
                free(ck);
            }
        }
        ck = cfile->curr_chunk;
        if (ck) {               //  free the curr_active chunk
            error(" Curr_chunk not finished...\n");
            dump_chunk(ck);
            free_buf_slot(hca.rdma_buf, ck->bufid, 0);
            free(ck);
        }
        //pthread_mutex_unlock( &cfile->mutex );
    } else                      // at ckpt-fs(client side), only with write-aggre
    {
        ck = cfile->curr_chunk;
        if (ck)                 // this should never happen!!!!
        {
            printf("%s (%s): Error!! Find a leftover chunk!!\n", __func__, path);
            printf("\t%s: %s: flush final chunk: (%ld@%ld)\n", __func__, cfile->filename, ck->curr_pos, ck->offset);
            ck->is_last_chunk = 1;
            cfile->adv_size += ck->curr_pos;
            atomic_inc(&minfo.chunk_cnt);
            workqueue_enqueue(iopool->queue, ck, sizeof(*ck), 0, 0);
            free(ck);
            cfile->curr_chunk = NULL;
        }
    }

    cfile->can_release = 1;
    if (hash_table_put_record(g_ht_cfile, cfile, 1) > 0) {
        dbg(" [%s] fi=%p, set fi->fh to NULL...\n", cfile->filename, fi);
        fi->fh = 0;
    }
    htbl_unlock_bucket(bkt);
    /**
    strncpy(filename, cfile->filename, 128);
    filename[128-1] = 0;
    //pthread_mutex_lock( &cfile->mutex );
    cfile->can_release = 1;    
    //pthread_mutex_unlock( &cfile->mutex );
    /// after setting "can_release", 
    /// an iothr may have cleaned the hash-tbl to remove this file
    hash_bucket_t* bkt = htbl_find_lock_bucket(g_ht_cfile, filename, strlen(filename));
    if( bkt )
    {    // has located the cfile-record, and locked the hosting hash-bucket
        if( hash_table_put_record( g_ht_cfile, cfile, 1)>0 )
        {
            //dbg(" [%s] fi=%p, set fi->fh to NULL...\n", cfile->filename, fi );
            fi->fh = NULL;
        }
        htbl_unlock_bucket(bkt);
    }
    else{
        dbg(" other thr has del cfile-record for (%s)\n", filename);
    } **/

    // dump_hash_table( g_ht_cfile ); //// don't call this: will hang...
    //printf("At %s:: dump ib-buf...\n", __func__);
    //dump_ib_buffer( hca.rdma_buf );

#endif
    ///////////////////////////////////////////////////////////

    return retstat;
}

/**
both (path) and (newpath) are relative to the root of mnt-point.
Need to turn them to full-path
**/
int crfs_rename(const char *path, const char *newpath)
{
    int retstat = 0;
    char fpath[PATH_MAX];
    char fpath_new[PATH_MAX];   // full-path

//    log_msg("\ncrfs_rename(fpath=\"%s\", newpath=\"%s\")\n", path, newpath);
    dbg("\ncrfs_rename(fpath=\"%s\", newpath=\"%s\")\n", path, newpath);
    //dbg(" mv %s to %s\n", path, newpath );

    /////////////////////////////////////////////////
#ifdef CKPT_MIG
    ckpt_file_t *cfile = hash_table_get_record(g_ht_cfile, path, 0);
    if (!cfile)                 // not a ckpt-file, fallback to normal path
    {
        //error("Error: ckptfile %s: cfile is NULL...\n", path);
        //return -EIO;
    }
    // if( cfile && crfs_mode == CRFS_CLIENT ) /// at ckpt-fs(client-side):
    if (cfile && crfs_mode == MODE_WRITEAGGRE)  /// at ckpt-fs(client-side):  
    {                           // write-aggre may still have some data in curr-chunk, flush it 

        ckpt_chunk_t *chunk = cfile->curr_chunk;
        if (chunk) {
            dbg("\t%s: flush final chunk before rename\n", cfile->filename);
            chunk->is_last_chunk = 1;
            cfile->adv_size += chunk->curr_pos;
            atomic_inc(&minfo.chunk_cnt);
            workqueue_enqueue(iopool->queue, chunk, sizeof(*chunk), 0, 0);
            /////retstat = pwrite(cfile->fd, chunk->buf, chunk->curr_pos, chunk->offset);
            free(chunk);
            cfile->curr_chunk = NULL;
        }
    }
#endif

    crfs_fullpath(fpath, path);
    crfs_fullpath(fpath_new, newpath);

    retstat = rename(fpath, fpath_new);
    if (retstat < 0)
        retstat = crfs_error("crfs_rename rename");

    return retstat;
}

/** Get file system statistics 
 * The 'f_frsize', 'f_favail', 'f_fsid' and 'f_flag' fields are ignored 
 * Replaced 'struct statfs' parameter with 'struct statvfs' in
 * version 2.5
*/
int crfs_statfs(const char *path, struct statvfs *statv)
{
    int retstat = 0;
    char fpath[PATH_MAX];

//    log_msg("\ncrfs_statfs(path=\"%s\", statv=0x%08x)\n", path, statv);
    dbg("\ncrfs_statfs(path=\"%s\", statv=0x%08x)\n", path, statv);
    crfs_fullpath(fpath, path);

    // get stats for underlying filesystem
    retstat = statvfs(fpath, statv);
    if (retstat < 0)
        retstat = crfs_error("crfs_statfs statvfs");

    log_statvfs(statv);

    return retstat;
}

/** Remove a file */
////////// NOTE: needed by BLCR!!
int crfs_unlink(const char *path)
{
    int retstat = 0;
    char fpath[PATH_MAX];

//    log_msg("crfs_unlink(path=\"%s\")\n", path);
    dbg("crfs_unlink(path=\"%s\")\n", path);
    crfs_fullpath(fpath, path);

    retstat = unlink(fpath);
    if (retstat < 0)
        retstat = crfs_error("crfs_unlink unlink");

    return retstat;
}

/**
Change the size of a file. 
NOTE:: This func is needed if want ot overwrite an existing file.
**/
int crfs_truncate(const char *path, off_t newsize)
{
    int retstat = 0;
    char fpath[PATH_MAX];

    dbg("\t(path=\"%s\", newsize=%lld)\n", path, newsize);
    crfs_fullpath(fpath, path);

    retstat = truncate(fpath, newsize);
    if (retstat < 0)
        error("truncate failed...\n");
    dbg("ret = %d\n", retstat);
    return retstat;
}

/**
Set extended attributes.
At ckpt-fs(client) side, set the "target of migration", so client can connect to it.
Usually the "path" can be simply set as "/".
Or, the "path" must be an existing file in the FS.
**/
int crfs_setxattr(const char *path, const char *name, const char *value, size_t size, int flags)
{
    int retstat = 0;
    //char fpath[PATH_MAX];
    //crfs_fullpath(fpath, path);

    dbg("\n%s(path=\"%s\", name=\"%s\", value=\"%s\", size=%d, flags=0x%08x)\n", __func__, path, name, value, size, flags);

    if (strcmp(name, "migration.src") == 0) //set mig-src
    {
        if (size >= MAX_HOSTNAME_LENGTH) {
            err("mig-src too long: %s\n", value);
            return -1;
        }
        strncpy(minfo.src, value, size);
        minfo.src[size] = 0;
    } else if (strcmp(name, "migration.tgt") == 0)  // set mig-tgt
    {
        if (size >= MAX_HOSTNAME_LENGTH) {
            err("mig-tgt too long: %s\n", value);
            return -1;
        }
        strncpy(minfo.tgt, value, size);
        minfo.tgt[size] = 0;
    } else if (strcmp(name, "migration.state") == 0)    // a migration is finished
    {
        int ti = *((int *) value);
        if (ti == 1)            // want to start a mig
        {
            if (mig_state == 1) {   // an active migration is going on,
                err("Already has a migration going on. ret...\n");
                return 0;
            } else if (mig_role == ROLE_MIG_SRC) {
                sem_init(&minfo.sem, 0, 0);

                atomic_set(&minfo.chunk_cnt, 0);
                sem_init(&minfo.chunk_comp_sem, 0, 0);

                if (ibcli_start_mig(&minfo) != 0) {
                    err("start migration failed...\n");
                    mig_state = 0;
                    retstat = -1;
                } else
                    mig_state = 1;
            } else
                mig_state = 1;
        } else                  // want to terminate a mig
        {
            if (mig_state == 1 && mig_role == ROLE_MIG_SRC) {   // has an active mig
                mig_state = 0;
                ibcli_end_mig(&minfo);
                memset(&minfo, 0, sizeof(minfo));
            } else
                mig_state = 0;
        }
    } else {
        dbg("Unknown name=%s, value=%s\n", name, value);
        retstat = lsetxattr(path, name, value, size, flags);
    }

    dbg("mig-src=%s, tgt=%s, mig_state=%d\n", minfo.src, minfo.tgt, mig_state);

    return retstat;
}

/** 
Get extended attributes.
The "path" must be an existing file in the FS, or it is "/"
**/
int crfs_getxattr(const char *path, const char *name, char *value, size_t size)
{
    int retstat = 0;
    //char fpath[PATH_MAX];
    //crfs_fullpath(fpath, path);

    dbg("\n%s(path = \"%s\", name = \"%s\", value = \"%s\", size = %d)\n", __func__, path, name, value, size);

    if (strcmp(name, "migration.src") == 0) //set mig-src
    {
        retstat = strlen(minfo.src);
        memcpy(value, minfo.src, retstat);
    } else if (strcmp(name, "migration.tgt") == 0)  // set mig-tgt
    {
        retstat = strlen(minfo.tgt);
        memcpy(value, minfo.tgt, retstat);
    } else if (strcmp(name, "migration.state") == 0)    // a migration is finished
    {
        //snprintf(value, "%d", has_active_mig );
        memcpy(value, &mig_state, sizeof(int));
        retstat = sizeof(int);
    } else {
        retstat = lgetxattr(path, name, value, size);
    }

    return retstat;
}

///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct fuse_operations crfs_oper = {

    .init = crfs_init,
    .destroy = crfs_destroy,

    .getattr = crfs_getattr,

    .flush = crfs_flush,        // when app calls close(), translates to: flush(); then release()
    .release = crfs_release,

    /// dir-ops
    .opendir = crfs_opendir,
    .readdir = crfs_readdir,
    .releasedir = crfs_releasedir,
    .fsyncdir = crfs_fsyncdir,
    .mkdir = crfs_mkdir,
    .rmdir = crfs_rmdir,

    /// file-ops
    .create = crfs_create,
    .open = crfs_open,
    .read = crfs_read,
    .write = crfs_write,
    .unlink = crfs_unlink,
    .fsync = crfs_fsync,
    .rename = crfs_rename,      // needed by BLCR

    .truncate = crfs_truncate,

    .setxattr = crfs_setxattr,
    .getxattr = crfs_getxattr,

};

void crfs_usage()
{
    fprintf(stderr, "Usage: crfs  [real-work-dir] [mount-Point] <-opt1> <-opt2> ");
}

int crfs_main(int pfd, int argc, char *argv[])
{
    int i;
    int fuse_stat;
    struct crfs_state *crfs_data;

    ////////// check the params:   ./crfs [real-work-dir] [mount-Point] <-opt1> <-opt2> 
    if (argc < 3) {
        crfs_usage();
        return -1;
    }
#if BUILDIN_MOD
    pipe_fd = pfd;
#endif

    for (i = 1; (i < argc) && (argv[i][0] == '-'); i++) ;

    if (i == argc) {
        crfs_usage();
        return -1;
    }

    crfs_data = calloc(sizeof(struct crfs_state), 1);
    if (crfs_data == NULL) {
        perror("main calloc");
        return -1;
    }
    crfs_data->rootdir = realpath(argv[i], NULL);

    for (; i < argc; i++)
        argv[i] = argv[i + 1];
    argc--;

    //crfs_data->logfile = log_open("crfslog");
    crfs_data->logfile = NULL;  // don't use log for now

    dbg("\n*************\nabout to call fuse_main, crfs_data = %p....\n", crfs_data);
    fuse_stat = fuse_main(argc, argv, &crfs_oper, crfs_data);
    // inside fuse_main(), init the fuse_context_private = crfs_data, 

    /// when fusermount -u , fuse_destroy() is called, and then, comes here:
    dbg("\n*************\nfuse_main returned %d.... \n", fuse_stat);

    if (crfs_data->logfile)
        fclose(crfs_data->logfile);
    free(crfs_data);

    return fuse_stat;
}

#endif
