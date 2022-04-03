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

#ifndef CKPT_FILE_H
#define CKPT_FILE_H

#include <sys/time.h>
#include "common_crfs.h"
#include "atomic.h"
#include "list.h"

/////////////////////////////

#define    HASH_FROM_FILENAME  1    // if this is defined, ckpt-files are hashed to htable using its file-name;
    /// otherwise, hash using the (ckpt-id, proc-id)

///////////////////////////
struct ckpt_file_s;

typedef struct ckpt_chunk_s {
    int ckpt_id;                // id of this ckpt
    int proc_rank;              // proc rank this file belongs to

    struct ckpt_file_s *ckpt_file;

    unsigned long offset;       // offset of this chunk in orig ckpt file
    unsigned long size;         // size of this chunk

    void *buf;                  // buf addr where this chunk is stored
    int bufid;                  // buf-id in the ib_buffer
    unsigned long curr_pos;     // currently read/write at this pos in buf

    int is_last_chunk;          // Is this chunk the last chunk of the file??

    struct list_head list;      // link to all chunks belonging to a ckpt file (ckptid, proc_rank)
    // this list is in ascending order of offset
} ckpt_chunk_t;

/// ckpt-file with many chunks,  linked into a openhash-bucket
typedef struct ckpt_file_s {
    int ckpt_id;                // (ckpt-id, proc_rank) of this ckpt-file
    int proc_rank;

    int fd;                     // backend file-descriptor of this ckpt-file

    pthread_mutex_t mutex;      // serialize io-threads writing to same ckpt-file

    pthread_mutex_t io_mtx;     //only 1 io-thr can write to this file at given time

    /////////// obsolete fields
    atomic_t chunk_to_write;    // num of pending RR-rqst that have been received 
    // Inc when a RR rqst comse, Dec when a RR completes
    int chunk_num;              // num of chunks have been RRed for this ckpt file  
    ///////////////////////////////////////

    int can_release;            // Indicate: release() is called, and can release this struct if last_chunk is finished   
    int has_last_chunk;         // indicate if the last chunk of this file is received
    unsigned long write_size;   // total data that have been written to this cfile

    unsigned long adv_size;     // advertised size of this file. Whenever a chunk is posted, adv_size is increased

    atomic_t ref;               ///// num of references
    // ref-inc: open this ckpt-file , add a new RR-rqst
    // ref-dec: finish a RR-rqst,  close this ckpt-file

    struct list_head list;      // link to next ckpt-files in a same hash-bucket

    int rcv_pending_RR_num;     // number of RR going on (pending to be completed)
    int rcv_ready_chunk_num;    // num of chunks that have been RRed, ready to use
    int rcv_total_chunk_num;    // total num of chunks for this cfile, including:
    // pending RR +  ready-chunks + chunks-already-read

    struct list_head chunk_list;    // for recv: link all ready ckpt-chunks belongint to this ckptfile
    /// the list in ascending order of offset
    ckpt_chunk_t *curr_chunk;   // the buf-chunk currently being used.
    /// chunks are used in ascending order of offset

    sem_t sem;                  // for recv: those who want to read data wait on this sem, for RR to fetch the chunk;

    ////////
    int filename_len;
    char filename[MAX_FILENAME_LENGTH]; // ckpt-file name
    ///////////////

    struct timeval tstart;      // time when this chunk-file is open/finished
    struct timeval tend;
    // for sender: those who has data to write wait on this sem, for a buf-chunk to be avail
} ckpt_file_t;

//////////////////////

ckpt_file_t *alloc_ckpt_file();
ckpt_chunk_t *alloc_ckpt_chunk();

int free_ckpt_file(ckpt_file_t * cfile);

int add_chunk_to_ckpt_file(ckpt_file_t * cfile, struct ib_packet *rrpkt);

//void    dump_chunk_list(ckpt_file_t* cfile );
void dump_ckpt_file(ckpt_file_t * cfile);
void dump_chunk(ckpt_chunk_t * ck);

ckpt_chunk_t *get_chunk_from_ckpt_file(ckpt_file_t * cfile, int offset);

void ckpt_free_chunk(ckpt_chunk_t * chunk);

#endif                          //CKPT_FILE_H
