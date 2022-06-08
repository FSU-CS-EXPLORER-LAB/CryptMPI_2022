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

#include "pmi_tree.h"
#include "mpispawn_tree.h"
#include "mpirun_util.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <netdb.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/sendfile.h>
#include "mpichconf.h"
#include "debug_utils.h"
#include <mpispawn_error_codes.h>

#if defined(CKPT) && defined(CR_FTB)
#include "mpispawn_ckpt.h"
#endif

// Defined in ...
extern int mt_id;
extern process_info_t *local_processes;
extern child_t *children;
extern int NCHILD;
extern int N;
extern int MPISPAWN_HAS_PARENT;
extern int MPISPAWN_NCHILD;
extern int *mpispawn_fds;

static fd_set child_socks;
static int NCHILD_INCL;

#ifdef CR_DEBUG
#define CR_DBG(args...)  do {\
  fprintf(stderr, "\t [mtid %d][%s: line %d]", mt_id,__FILE__, __LINE__);    \
  fprintf(stderr, args);\
 }while(0)
#else
#define CR_DBG(args...)
#endif

/* #define dbg(fmt, args...)  do{ \
    fprintf(stderr,"%s: [mt_id %d]: "fmt, __func__, mt_id, ##args);fflush(stderr);} while(0) */
#define dbg(fmt, args...)

/* list of pending requests that we've sent elsewhere. Change to a hash table 
 * when needed */

req_list_t *pending_req_head = NULL;
req_list_t *pending_req_tail = NULL;

kv_cache_t *kv_cache[KVC_HASH_SIZE];
kv_cache_t *kv_pending_puts;

static int npending_puts;

char *handle_spawn_request(int fd, char *buf, int buflen);

int get_req_dest(int req_rank, char **key)
{
    req_list_t *iter = pending_req_head;
    int ret_fd;

    while (iter != NULL) {
        if (iter->req_rank == req_rank) {
            ret_fd = iter->req_src_fd;

            if (iter->req_prev)
                iter->req_prev->req_next = iter->req_next;
            else
                pending_req_head = iter->req_next;
            if (iter->req_next)
                iter->req_next->req_prev = iter->req_prev;
            else
                pending_req_tail = iter->req_prev;
            if (iter->req_key) {
                *key = iter->req_key;
            }
            free(iter);
            return ret_fd;
        }
        iter = iter->req_next;
    }
    mpispawn_abort(ERR_REQ);
    return -1;
}

int save_pending_req(int req_rank, char *req_key, int req_fd)
{
    req_list_t *preq = (req_list_t *) malloc(req_list_s);

    if (!preq) {
        mpispawn_abort(ERR_MEM);
        return -1;
    }
    if (req_key) {
        preq->req_key = (char *) malloc((strlen(req_key) + 1) * sizeof(char));
        if (!preq->req_key) {
            mpispawn_abort(ERR_MEM);
            return -1;
        }
        strcpy(preq->req_key, req_key);
        preq->req_key[strlen(req_key)] = 0;
    } else
        preq->req_key = NULL;
    preq->req_rank = req_rank;
    preq->req_src_fd = req_fd;
    preq->req_prev = pending_req_tail;
    if (pending_req_tail != NULL)
        pending_req_tail->req_next = preq;
    pending_req_tail = preq;

    preq->req_next = NULL;
    if (pending_req_head == NULL)
        pending_req_head = preq;

    return 0;
}

unsigned int kvc_hash(char *s)
{

    unsigned int hash = 0;
    while (*s)
        hash ^= *s++;
    return hash & KVC_HASH_MASK;
}

void delete_kvc(char *key)
{
    kv_cache_t *iter, *prev;
    unsigned int hash = kvc_hash(key);

    prev = iter = kv_cache[hash];

    while (NULL != iter) {
        if (!strcmp(iter->kvc_key, key)) {
            if (iter == kv_cache[hash]) {
                kv_cache[hash] = iter->kvc_hash_next;
            } else {
                prev->kvc_hash_next = iter->kvc_hash_next;
            }
            free(iter->kvc_val);
            free(iter->kvc_key);
            free(iter);
            return;
        }
        prev = iter;
        iter = iter->kvc_hash_next;
    }
}

int add_kvc(char *key, char *val, int from_parent)
{
    kv_cache_t *pkvc;
    unsigned int hash = kvc_hash(key);
    pkvc = (kv_cache_t *) malloc(kv_cache_s);
    if (!pkvc) {
        mpispawn_abort(ERR_MEM);
        return -1;
    }

    pkvc->kvc_key = (char *) malloc((strlen(key) + 1) * sizeof(char));
    pkvc->kvc_val = (char *) malloc((strlen(val) + 1) * sizeof(char));;
    if (!pkvc->kvc_key || !pkvc->kvc_val) {
        mpispawn_abort(ERR_MEM);
        return -1;
    }
    strcpy(pkvc->kvc_key, key);
    strcpy(pkvc->kvc_val, val);
    if (val[strlen(val) - 1] == '\n')
        pkvc->kvc_val[strlen(val) - 1] = 0;
    pkvc->kvc_val[strlen(val)] = 0;
    pkvc->kvc_key[strlen(key)] = 0;
    pkvc->kvc_hash_next = NULL;

    kv_cache_t *iter = kv_cache[hash];

    if (NULL == iter) {
        kv_cache[hash] = pkvc;
    } else {
        pkvc->kvc_hash_next = kv_cache[hash];
        kv_cache[hash] = pkvc;
    }
    if (!from_parent) {
        pkvc->kvc_list_next = kv_pending_puts;
        kv_pending_puts = pkvc;
        npending_puts++;
    }
    return 0;
}

char *check_kvc(char *key)
{
    kv_cache_t *iter;
    unsigned int hash = kvc_hash(key);

    iter = kv_cache[hash];

    while (NULL != iter) {
        if (!strcmp(iter->kvc_key, key)) {
            return iter->kvc_val;
        }
        iter = iter->kvc_hash_next;
    }
    return NULL;
}

int clear_kvc(void)
{
    int i;
    kv_cache_t *iter, *tmp;

    for (i = 0; i < KVC_HASH_SIZE; i++) {
        iter = kv_cache[i];
        while (iter) {
            tmp = iter;
            iter = iter->kvc_hash_next;
            free(tmp->kvc_key);
            free(tmp->kvc_val);
            free(tmp);
        }
        kv_cache[i] = 0;
    }
    return 0;
}

int writeline(int fd, char *msg, int msglen)
{
    int n;
    MT_ASSERT(msg[msglen - 1] == '\n');

    do {
        n = write(fd, msg, msglen);
    }
    while (n == -1 && errno == EINTR);

    if (n < 0) {
        PRINT_ERROR_ERRNO("write() failed", errno);
        mpispawn_abort(MPISPAWN_PMI_WRITE_ERROR);
    } else if (n < msglen) {
        PRINT_ERROR("write() failed (msglen = %d, written size = %d)\n", msglen, n);
        mpispawn_abort(MPISPAWN_PMI_WRITE_ERROR);
    }
    return n;
}

int read_size(int fd, void *msg, int size)
{
    int n = 0, rc;
    char *offset = (char *) msg;

    while (n < size) {
        do {
            rc = read(fd, offset, size - n);
        } while ( rc == -1 && (errno == EINTR || errno == EAGAIN) );

        if (rc < 0) {
            PRINT_ERROR_ERRNO("read() failed on file descriptor %d", errno, fd);
            return rc;
        } else if (0 == rc) {
#if defined(CKPT) && defined(CR_FTB)
            if ( !cr_mig_src ) {
#endif
                PRINT_ERROR("Unexpected End-Of-File on file descriptor %d. MPI process died?\n", fd);
#if defined(CKPT) && defined(CR_FTB)
            }
#endif
            return n;
        }
        offset += rc;
        n += rc;
    }
    return n;
}

int write_size(int fd, void *msg, int size)
{
    int rc, n = 0;
    char *offset = (char *) msg;

    while (n < size) {
        rc = write(fd, offset, size - n);

        if (rc < 0) {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            return rc;
        } else if (0 == rc)
            return n;

        offset += rc;
        n += rc;
    }
    return n;
}

int readline(int fd, char *msg, int maxlen)
{
    int n;
    MT_ASSERT(maxlen == MAXLINE);

    do {
        n = read(fd, msg, maxlen);
    } while (n == -1 && errno == EINTR);

    if (n < 0) {
        PRINT_ERROR_ERRNO("read() failed on file descriptor %d", errno, fd);
        return n;
    } else if ( n == 0 ) {
#if defined(CKPT) && defined(CR_FTB)
        if ( !cr_mig_src ) {
#endif
            PRINT_ERROR("Unexpected End-Of-File on file descriptor %d. MPI process died?\n", fd);
#if defined(CKPT) && defined(CR_FTB)
        }
#endif
        return 0;
    }
    if (n < MAXLINE) {
        msg[n] = '\0';
    }

    MT_ASSERT(n <= MAXLINE);
    MT_ASSERT(msg[n - 1] == '\n');
    return n;
}

/* send_parent
 * src: -1 Propagate put message 
 *       n Propagate get request from rank n */

int send_parent(int src, char *msg, int msg_len)
{
    msg_hdr_t hdr = { src, msg_len, -1 };
    write(MPISPAWN_PARENT_FD, &hdr, msg_hdr_s); // new
    writeline(MPISPAWN_PARENT_FD, msg, msg_len);
    return 0;
}

/*
#define CHECK(s1, s2, dst) if (strcmp(s1, s2) == 0) { \
    MT_ASSERT(end-start); \
    dst = (char *) malloc (sizeof (char) * (end-start + 1)); \
    if (!dst) { \
        rv = ERR_MEM; \
        goto exit_err; \
    } \
    strncpy (dst, start, end-start); \
    dst[end-start] = 0; \
}
*/

/**
 * Some message is not sent by all local nodes, in those cases the aggregation fails.
 * Some other message may need to be delivered soon. 
 * We check for those messages and we send them without aggregating.
 */
static char *flushing_messages[] = {
    "sharedFilename",
    "MV2BUF",
    NULL
};

int check_pending_puts(void)
{
    kv_cache_t *iter, *tmp;
    msg_hdr_t hdr = { -1, -1, MT_MSG_BPUTS };
    char *buf, *pbuf;
    int i;
    int msg_flush = 0;

    /* Check if one of the messages in the pending list is listed in flushing_messages.
       If so, we send them now */
    iter = kv_pending_puts;
    while (iter && !msg_flush) {
        i = 0;
        while ((flushing_messages[i] != NULL) && (!msg_flush)) {
            if (strncmp(iter->kvc_key, flushing_messages[i], strlen(flushing_messages[i])) == 0) {
                msg_flush = 1;
            }

            i++;
        }
        iter = iter->kvc_list_next;
    }

    if ((!msg_flush) && (npending_puts != NCHILD + NCHILD_INCL)) {
        return 0;
    }
#define REC_SIZE (KVS_MAX_KEY + KVS_MAX_VAL + 2)
    hdr.msg_len = REC_SIZE * npending_puts + 1;
    buf = (char *) malloc(hdr.msg_len * sizeof(char));
    pbuf = buf;
    iter = kv_pending_puts;
    while (iter) {
        snprintf(pbuf, KVS_MAX_KEY, "%s", iter->kvc_key);
        pbuf[KVS_MAX_KEY] = 0;
        pbuf += KVS_MAX_KEY + 1;
        snprintf(pbuf, KVS_MAX_VAL, "%s", iter->kvc_val);
        pbuf[KVS_MAX_VAL] = 0;
        pbuf += KVS_MAX_VAL + 1;

        tmp = iter->kvc_list_next;
        iter->kvc_list_next = NULL;
        iter = tmp;

        npending_puts--;
    }
    MT_ASSERT(npending_puts == 0);
    kv_pending_puts = NULL;
#undef REC_SIZE

    buf[hdr.msg_len - 1] = '\n';
    if (MPISPAWN_HAS_PARENT) {
        write(MPISPAWN_PARENT_FD, &hdr, msg_hdr_s);
        write_size(MPISPAWN_PARENT_FD, buf, hdr.msg_len);
    } else {
        /* If I'm root, send it down the tree */
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            write(MPISPAWN_CHILD_FDS[i], &hdr, msg_hdr_s);
            write_size(MPISPAWN_CHILD_FDS[i], buf, hdr.msg_len);
        }
    }
    free(buf);
    return 0;
}

static char *resbuf = "cmd=spawn_result rc=0\n";

int parse_str(int rank, int fd, char *msg, int msg_len, int src)
{
    dbg(": rank=%d,fd=%d, msg:%s:msg_len=%d, src=%d\n", rank, fd, msg, msg_len, src);
    static int barrier_count;
    int rv = 0, i;
    char *p = msg, *start = NULL, *end = NULL;
    char *command = NULL, *key = NULL, *val = NULL, *pmi_version = NULL, *pmi_subversion = NULL, *kvsname = NULL, *rc = NULL, *pmiid = NULL;
    char name[KVS_MAX_NAME];
    char resp[MAXLINE];
    char *kvstmplate;

    msg_hdr_t hdr;

    if (!p)
        return -1;

    start = p;
    while (*p != '\n') {
        if (*p == '=') {
            end = p;
            strncpy(name, start, end - start);
            name[end - start] = 0;
            p++;
            start = p;
            while (*p != ' ' && *p != '\n') {
                p++;
            }
            end = p;
            switch (strlen(name)) {
            case 3:            /* cmd, key */
                /* CHECK (name, "cmd", command) */
                if (strcmp(name, "cmd") == 0) {
                    MT_ASSERT(end - start);
                    command = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!command) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(command, start, end - start);
                    command[end - start] = 0;
                } else
                    /* CHECK (name, "key", key) */
                if (strcmp(name, "key") == 0) {
                    MT_ASSERT(end - start);
                    key = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!key) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(key, start, end - start);
                    key[end - start] = 0;
                }
                break;
            case 4:
                /* CHECK (name, "mcmd", command) */
                if (strcmp(name, "mcmd") == 0) {
                    MT_ASSERT(end - start);
                    command = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!command) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(command, start, end - start);
                    command[end - start] = 0;
                } else
                    /* CHECK (name, "port", val) */
                if (strcmp(name, "port") == 0) {
                    MT_ASSERT(end - start);
                    val = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!val) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(val, start, end - start);
                    val[end - start] = 0;
                }
                break;
            case 7:            /* kvsname */
                /* CHECK (name, "kvsname", kvsname) */
                if (strcmp(name, "kvsname") == 0) {
                    MT_ASSERT(end - start);
                    kvsname = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!kvsname) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(kvsname, start, end - start);
                    kvsname[end - start] = 0;
                } else
                    /* CHECK (name, "service", key) */
                if (strcmp(name, "service") == 0) {
                    MT_ASSERT(end - start);
                    key = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!key) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(key, start, end - start);
                    key[end - start] = 0;
                }
                break;
            case 5:            /* value, pmiid */
                /* CHECK (name, "value", val) */
                if (strcmp(name, "value") == 0) {
                    MT_ASSERT(end - start);
                    val = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!val) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(val, start, end - start);
                    val[end - start] = 0;
                } else
                    /* CHECK (name, "pmiid", pmiid) */
                if (strcmp(name, "pmiid") == 0) {
                    MT_ASSERT(end - start);
                    pmiid = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!pmiid) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(pmiid, start, end - start);
                    pmiid[end - start] = 0;
                }
                break;
            case 11:           /* pmi_version */
                /* CHECK (name, "pmi_version", pmi_version) */
                if (strcmp(name, "pmi_version") == 0) {
                    MT_ASSERT(end - start);
                    pmi_version = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!pmi_version) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(pmi_version, start, end - start);
                    pmi_version[end - start] = 0;
                }
                break;
            case 14:           /* pmi_subversion */
                /* CHECK (name, "pmi_subversion", pmi_subversion) */
                if (strcmp(name, "pmi_subversion") == 0) {
                    MT_ASSERT(end - start);
                    pmi_subversion = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!pmi_subversion) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(pmi_subversion, start, end - start);
                    pmi_subversion[end - start] = 0;
                }
                break;
            case 2:            /* rc */
                /* CHECK (name, "rc", rc) */
                if (strcmp(name, "rc") == 0) {
                    MT_ASSERT(end - start);
                    rc = (char *) malloc(sizeof(char) * (end - start + 1));
                    if (!rc) {
                        rv = ERR_MEM;
                        goto exit_err;
                    }
                    strncpy(rc, start, end - start);
                    rc[end - start] = 0;
                }
                break;
            default:
                rv = ERR_STR;
                break;
            }
            if (*p != '\n') {
                start = ++p;
            }
        }
        if (*p != '\n')
            p++;
    }

    CR_DBG(">parse_str(v): command:%s:\n", command);
    switch (strlen(command)) {
    case 3:                    /* get, put */
        if (0 == strcmp(command, "get")) {
            char *kvc_val = check_kvc(key);
            hdr.msg_rank = rank;
            if (kvc_val) {
                sprintf(resp, "cmd=get_result rc=0 value=%s\n", kvc_val);
                dbg(" cmd=get, key=%s, find val=%s\n", key, kvc_val);
            } else {
                sprintf(resp, "cmd=get_result rc=1 value=%s\n", "NOTFOUND");
                dbg(" ****  ERROR:: PMI key '%s' not found.\n", key);
            }

            hdr.msg_len = strlen(resp);
            if (src == MT_CHILD) {
                write(fd, &hdr, msg_hdr_s);
            }
            writeline(fd, resp, hdr.msg_len);
        }
        /* cmd=put */
        else if (0 == strcmp(command, "put")) {
            hdr.msg_rank = rank;
            hdr.msg_len = msg_len;
            add_kvc(key, val, 0);
            check_pending_puts();
            if (src == MT_RANK) {
                sprintf(resp, "cmd=put_result rc=0\n");
                writeline(fd, resp, strlen(resp));
            }
        } else
            goto invalid_cmd;
        break;
    case 4:                    /* init */
        if (0 == strcmp(command, "init")) {
            if (pmi_version[0] == PMI_VERSION && pmi_version[1] == '\0' && pmi_subversion[0] == PMI_SUBVERSION && pmi_subversion[1] == '\0') {
                sprintf(resp, "cmd=response_to_init pmi_version=%c " "pmi_subversion=%c rc=0\n", PMI_VERSION, PMI_SUBVERSION);
                writeline(fd, resp, strlen(resp));
            } else {
                sprintf(resp, "cmd=response_to_init pmi_version=%c " "pmi_subversion=%c rc=1\n", PMI_VERSION, PMI_SUBVERSION);
                writeline(fd, resp, strlen(resp));
            }
        } else
            goto invalid_cmd;
        break;
    case 5:                    /* spawn */
        if (0 == strcmp(command, "spawn")) {
            handle_spawn_request(fd, msg, msg_len);
            /* send response to spawn request */
            write(fd, resbuf, strlen(resbuf));
        }
        break;
    case 7:                    /* initack */
        if (0 == strcmp(command, "initack")) {
            CR_DBG("> parse_str()command = initack\n");
            dbg("*** initack: NCHILD=%d\n", NCHILD);
            for (i = 0; i < NCHILD; i++) {
                if (children[i].fd == fd) {
                    children[i].rank = atoi(pmiid);
                    /* TD validate rank */
                    goto initack;
                }
            }
            if (i == NCHILD) {
                dbg("*********** Error::  got initack for child-%d of %d, no match\n", i, NCHILD);
                rv = ERR_DEF;
                goto exit_err;
            }
          initack:
            sprintf(resp, "cmd=initack rc=0\ncmd=set size=%d\n" "cmd=set rank=%d\ncmd=set debug=0\n", N, children[i].rank);
            dbg(" reply initack: to fd=%d, with: %s\n", fd, resp);
            writeline(fd, resp, strlen(resp));
        }
        break;
    case 8:                    /* finalize */
        if (0 == strcmp(command, "finalize")) {
            barrier_count++;
#if defined(CKPT) && defined(CR_FTB)
            PRINT_DEBUG(DEBUG_FT_verbose, "barrier_count = %d, NCHILD = %d, MPISPAWN_NCHILD = %d, exclude_spare = %d\n", barrier_count, NCHILD, MPISPAWN_NCHILD, exclude_spare);
            if (barrier_count == (NCHILD + MPISPAWN_NCHILD - exclude_spare))
#else
            if (barrier_count == (NCHILD + MPISPAWN_NCHILD))
#endif
            {
                if (MPISPAWN_HAS_PARENT) {
                    send_parent(rank, msg, msg_len);
                    CR_DBG("[%d] send_parent rank %d \n", mt_id, rank);
                } else {
                    CR_DBG("[%d] goto finalize_ack \n", mt_id);
                    goto finalize_ack;
                }
            }
        } else
            goto invalid_cmd;
        break;
    case 9:                    /* get_maxes */
        if (0 == strcmp(command, "get_maxes")) {
            sprintf(resp, "cmd=maxes kvsname_max=%d keylen_max=%d " "vallen_max=%d\n", KVS_MAX_NAME, KVS_MAX_KEY, KVS_MAX_VAL);
            writeline(fd, resp, strlen(resp));
        } else
            goto invalid_cmd;
        break;
    case 10:                   /* get_appnum, get_result, put_result, barrier_in */
        if (0 == strcmp(command, "get_result")) {
            char *pkey;
            int child_fd;
            hdr.msg_rank = rank;
            hdr.msg_len = msg_len;
            child_fd = get_req_dest(rank, &pkey);
            free(pkey);
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                if (child_fd == MPISPAWN_CHILD_FDS[i]) {
                    write(child_fd, &hdr, msg_hdr_s);
                }
            }
            writeline(child_fd, msg, msg_len);
        } else if (0 == strcmp(command, "get_appnum")) {
            char *val;
            int multi, respval = 0;
            val = getenv("MPIRUN_COMM_MULTIPLE");
            if (val) {
                multi = atoi(val);
                if (multi)
                    respval = children[0].rank;
            }
            sprintf(resp, "cmd=appnum appnum=%d\n", respval);
            writeline(fd, resp, strlen(resp));
        } else if (0 == strcmp(command, "barrier_in")) {
            barrier_count++;
            char my_hostname[256];
            gethostname(my_hostname, 255);
#if defined(CKPT) && defined(CR_FTB)
            PRINT_DEBUG(DEBUG_FT_verbose, "MPISPAWN_HAS_PARENT = %d, barrier_count = %d, NCHILD = %d, MPISPAWN_NCHILD = %d, exclude_spare = %d\n", MPISPAWN_HAS_PARENT, barrier_count, NCHILD,
                        MPISPAWN_NCHILD, exclude_spare);
            if (barrier_count == (NCHILD + MPISPAWN_NCHILD - exclude_spare))
#else
            if (barrier_count == (NCHILD + MPISPAWN_NCHILD))
#endif
            {
                CR_DBG("[mt_id %d on %s] rank=%d MPIASPAWN_HAS_PARENT %d\n", mt_id, my_hostname, rank, MPISPAWN_HAS_PARENT);

                if (MPISPAWN_HAS_PARENT) {
                    /* msg_type */
                    CR_DBG("[mt_id %d on %s] ***rank %d SSEND TO PARENT \n", mt_id, my_hostname, rank);
                    send_parent(rank, msg, msg_len);
                } else {
                    CR_DBG("[mt_id %d on %s] ***rank %d GOTO barrier_out \n", mt_id, my_hostname, rank);
                    goto barrier_out;
                }
            }
        } else
            goto invalid_cmd;
        break;
    case 11:
        if (0 == strcmp(command, "barrier_out")) {
          barrier_out:
            {
                CR_DBG("[mt_id %d] ***rank %d barrier_out \n", mt_id, rank);
                sprintf(resp, "cmd=barrier_out\n");
                hdr.msg_rank = -1;
                hdr.msg_len = strlen(resp);
                hdr.msg_type = MT_MSG_BOUT;
                for (i = 0; i < MPISPAWN_NCHILD; i++) {
                    CR_DBG("[mt_id %d] ***rank %d barrier_out write %d\n", mt_id, rank, MPISPAWN_CHILD_FDS[i]);
                    write(MPISPAWN_CHILD_FDS[i], &hdr, msg_hdr_s);
                    writeline(MPISPAWN_CHILD_FDS[i], resp, hdr.msg_len);
                }
                for (i = 0; i < NCHILD; i++) {
                    writeline(children[i].fd, resp, hdr.msg_len);
                }
                barrier_count = 0;
                goto ret;
            }
        } else if (0 == strcmp(command, "lookup_name")) {
            char *valptr;
            valptr = check_kvc(key);
            if (valptr) {
                sprintf(resp, "cmd=lookup_result rc=0 port=%s\n", valptr);
                hdr.msg_len = strlen(resp);
                hdr.msg_rank = rank;
                if (src == MT_CHILD) {
                    write(fd, &hdr, msg_hdr_s);
                }
                writeline(fd, resp, strlen(resp));
            } else {
                if (MPISPAWN_HAS_PARENT) {
                    save_pending_req(rank, key, fd);
                    send_parent(rank, msg, msg_len);
                } else {
                    sprintf(resp, "cmd=lookup_result rc=1\n");
                    hdr.msg_len = strlen(resp);
                    hdr.msg_rank = rank;
                    if (src == MT_CHILD) {
                        write(fd, &hdr, msg_hdr_s);
                    }
                    writeline(fd, resp, strlen(resp));
                }
            }
            goto ret;
        } else
            goto invalid_cmd;
        break;
    case 12:                   /* finalize_ack */
        if (0 == strcmp(command, "finalize_ack")) {
            close(MPISPAWN_PARENT_FD);
          finalize_ack:
            {
                CR_DBG("[mt_id %d] ***rank %d finalize_ack write \n", mt_id, rank);
                hdr.msg_rank = -1;
                hdr.msg_type = MT_MSG_FACK;
                sprintf(resp, "cmd=finalize_ack\n");
                hdr.msg_len = strlen(resp);
                for (i = 0; i < MPISPAWN_NCHILD; i++) {
                    CR_DBG("[mt_id %d] ***rank %d finalize_ack write %d\n", mt_id, rank, MPISPAWN_CHILD_FDS[i]);
                    write(MPISPAWN_CHILD_FDS[i], &hdr, msg_hdr_s);
                    writeline(MPISPAWN_CHILD_FDS[i], resp, hdr.msg_len);
                    close(MPISPAWN_CHILD_FDS[i]);
                }
                for (i = 0; i < NCHILD; i++) {
                    writeline(children[i].fd, resp, hdr.msg_len);
                    close(children[i].fd);
                }
                barrier_count = 0;
                rv = 1;
                clear_kvc();
                goto ret;
            }
        } else if (0 == strcmp(command, "publish_name")) {
            add_kvc(key, val, 1);
            if (MPISPAWN_HAS_PARENT) {
                save_pending_req(rank, key, fd);
                send_parent(rank, msg, msg_len);
            } else {
                sprintf(resp, "cmd=publish_result rc=0\n");
                writeline(fd, resp, strlen(resp));
            }
            goto ret;
        } else
            goto invalid_cmd;
        break;
    case 13:
        if (0 == strcmp(command, "lookup_result")) {
            char *pkey;
            int tfd;
            hdr.msg_rank = rank;
            hdr.msg_len = msg_len;
            tfd = get_req_dest(rank, &pkey);
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                if (tfd == MPISPAWN_CHILD_FDS[i]) {
                    write(tfd, &hdr, msg_hdr_s);
                }
            }
            writeline(tfd, msg, msg_len);
            free(pkey);
            goto ret;
        } else
            goto invalid_cmd;
    case 14:                   /* get_my_kvsname */
        if (0 == strcmp(command, "get_my_kvsname")) {
            kvstmplate = getenv("MPDMAN_KVS_TEMPLATE");
            if (kvstmplate)
                sprintf(resp, "cmd=my_kvsname kvsname=%s_0\n", kvstmplate);
            else
                sprintf(resp, "cmd=my_kvsname kvsname=kvs_0\n");
            writeline(fd, resp, strlen(resp));
        } else if (0 == strcmp(command, "publish_result")) {
            int tfd;
            char *pkey;
            tfd = get_req_dest(rank, &pkey);
            hdr.msg_rank = rank;
            hdr.msg_len = msg_len;
            free(pkey);
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                if (tfd == MPISPAWN_CHILD_FDS[i]) {
                    write(tfd, &hdr, msg_hdr_s);
                }
            }
            writeline(tfd, msg, msg_len);
            goto ret;
        } else if (0 == strcmp(command, "unpublish_name")) {
            delete_kvc(key);
            if (MPISPAWN_HAS_PARENT) {
                save_pending_req(rank, key, fd);
                send_parent(rank, msg, msg_len);
            } else {
                sprintf(resp, "cmd=unpublish_result rc=0\n");
                writeline(fd, resp, strlen(resp));
            }
            goto ret;
        } else
            goto invalid_cmd;
        break;
    case 16:
        if (0 == strcmp(command, "unpublish_result")) {
            char *pkey;
            int tfd;
            hdr.msg_rank = rank;
            hdr.msg_len = msg_len;
            tfd = get_req_dest(rank, &pkey);
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                if (tfd == MPISPAWN_CHILD_FDS[i]) {
                    write(tfd, &hdr, msg_hdr_s);
                }
            }
            writeline(tfd, msg, msg_len);
            free(pkey);
            goto ret;
        } else
            goto invalid_cmd;
    case 17:                   /* get_universe_size */
        if (0 == strcmp(command, "get_universe_size")) {
            char *uni_size_str = NULL;
            int uni_size_value = -1; /* MPIR_UNIVERSE_SIZE_NOT_SET defined in mpiimpl.h */
            if ((uni_size_str = getenv("MV2_UNIVERSE_SIZE")) != NULL) {
                uni_size_value = atoi(uni_size_str);
                if (uni_size_value <= 0) {
                    uni_size_value = -1; /* MPIR_UNIVERSE_SIZE_NOT_SET defined in mpiimpl.h */
                }
            }

            sprintf(resp, "cmd=universe_size size=%d rc=0\n", uni_size_value);
            writeline(fd, resp, strlen(resp));
        } else
            goto invalid_cmd;
    }
    goto ret;

  invalid_cmd:
    printf("invalid %s\n", msg);
    rv = ERR_CMD;

  exit_err:
    mpispawn_abort(rv);

  ret:
    if (command != NULL)
        free(command);
    if (key != NULL)
        free(key);
    if (val != NULL)
        free(val);
    if (pmi_version != NULL)
        free(pmi_version);
    if (pmi_subversion != NULL)
        free(pmi_subversion);
    if (kvsname != NULL)
        free(kvsname);
    if (rc != NULL)
        free(rc);
    if (pmiid != NULL)
        free(pmiid);
    return rv;
}

int handle_mt_peer(int fd, msg_hdr_t * phdr)
{
    int rv = -1, n, i;
    char *buf = (char *) malloc(phdr->msg_len * sizeof(char));
    char *pkey, *pval;
    if (phdr->msg_type == MT_MSG_BPUTS) {
#define REC_SIZE (KVS_MAX_VAL + KVS_MAX_KEY + 2)
        if (read_size(fd, buf, phdr->msg_len) > 0) {
            if (MPISPAWN_HAS_PARENT && fd == MPISPAWN_PARENT_FD) {
                for (i = 0; i < MPISPAWN_NCHILD; i++) {
                    write(MPISPAWN_CHILD_FDS[i], phdr, msg_hdr_s);
                    write_size(MPISPAWN_CHILD_FDS[i], buf, phdr->msg_len);
                    dbg("xxx  BPUTS: write to child_%d: buf=%s\n", i, buf);
                }
            }
            n = (phdr->msg_len - 1) / REC_SIZE;
            for (i = 0; i < n; i++) {
                pkey = buf + i * REC_SIZE;
                pval = pkey + KVS_MAX_KEY + 1;
                dbg("xxxx BPUTS: add-kvc: buf=%s, key=%s, val=%s\n", buf, pkey, pval);
                add_kvc(pkey, pval, (MPISPAWN_HAS_PARENT && fd == MPISPAWN_PARENT_FD));
            }
            rv = 0;
        }
#undef REC_SIZE
        check_pending_puts();
    } else {
        int size = read_size(fd, buf, phdr->msg_len);
        if ( size !=  phdr->msg_len ) {
#if defined(CKPT) && defined(CR_FTB)
            if ( !cr_mig_src ) {
#endif
                PRINT_ERROR("Error while reading PMI socket. MPI process died?\n");
                mpispawn_abort(MPISPAWN_PMI_READ_ERROR);
#if defined(CKPT) && defined(CR_FTB)
            }
#endif
        }
        rv = parse_str(phdr->msg_rank, fd, buf, phdr->msg_len, MT_CHILD);
        if ( rv < 0) {
            PRINT_ERROR("parse_str() failed with error code %d\n", rv);
            mpispawn_abort(MPISPAWN_INTERNAL_ERROR);
        }
    }
    free(buf);
    return rv;
}

extern int mpirun_socket;

int mtpmi_init(void)
{
    int i, nchild_subtree = 0, tmp;
    int *children_subtree = (int *) malloc(sizeof(int) * MPISPAWN_NCHILD);

    for (i = 0; i < MPISPAWN_NCHILD; i++) {
        read_size(MPISPAWN_CHILD_FDS[i], &tmp, sizeof(int));
        children_subtree[i] = tmp;
        nchild_subtree += tmp;
    }
    NCHILD_INCL = nchild_subtree;
    nchild_subtree += NCHILD;
    if (MPISPAWN_HAS_PARENT)
        write(MPISPAWN_PARENT_FD, &nchild_subtree, sizeof(int));

    if (env2int("MPISPAWN_USE_TOTALVIEW") == 1) {
        process_info_t *all_pinfo;
        int iter = 0;
        if (MPISPAWN_NCHILD) {
            all_pinfo = (process_info_t *) malloc(process_info_s * nchild_subtree);
            if (!all_pinfo) {
                mpispawn_abort(ERR_MEM);
            }
            /* Read pid table from child MPISPAWNs */
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                read_socket(MPISPAWN_CHILD_FDS[i], &all_pinfo[iter], children_subtree[i] * process_info_s);
                iter += children_subtree[i];
            }
            for (i = 0; i < NCHILD; i++, iter++) {
                all_pinfo[iter].rank = local_processes[i].rank;
                all_pinfo[iter].pid = local_processes[i].pid;
            }
        } else {
            all_pinfo = local_processes;
        }

        if (MPISPAWN_HAS_PARENT) {
            write_socket(MPISPAWN_PARENT_FD, all_pinfo, nchild_subtree * process_info_s);
        } else if (mt_id == 0) {
            /* Send to mpirun_rsh */
            write_socket(mpirun_socket, all_pinfo, nchild_subtree * process_info_s);
            /* Wait for Totalview to be ready */
            read_socket(mpirun_socket, &tmp, sizeof(int));
            close(mpirun_socket);
        }
        /* Barrier */
        if (MPISPAWN_HAS_PARENT) {
            read_socket(MPISPAWN_PARENT_FD, &tmp, sizeof(int));
        }
        if (MPISPAWN_NCHILD) {
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                write_socket(MPISPAWN_CHILD_FDS[i], &tmp, sizeof(int));
            }
        }
    }
    return 0;
}

int mtpmi_processops(void)
{
    CR_DBG("mtpmi_process(-->v) \n");
    int ready, i, rv = 0;
    char buf[MAXLINE];
    msg_hdr_t hdr;
#if defined(CKPT) && defined(CR_FTB)
    int cleanup = 1;
#endif

    while (rv == 0) {
        FD_ZERO(&child_socks);
        if (MPISPAWN_HAS_PARENT)
            FD_SET(MPISPAWN_PARENT_FD, &child_socks);
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            FD_SET(MPISPAWN_CHILD_FDS[i], &child_socks);
        }
        for (i = 0; i < NCHILD; i++) {
            FD_SET(children[i].fd, &child_socks);
        }

#if defined(CKPT) && defined(CR_FTB)
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;
        //dbg("before select, has-par=%d, spawn-nchild=%d, nchild=%d\n", 
        //    MPISPAWN_HAS_PARENT, MPISPAWN_NCHILD, NCHILD );
        ready = select(FD_SETSIZE, &child_socks, NULL, NULL, &tv);
/*        if( ready>0 )
        dbg("select() ret ready=%d, cleanup=%d, cr_mig_tgt=%d, mig-src=%d\n", 
            ready, cleanup,cr_mig_tgt, cr_mig_src );*/
        if (ready == 0) {
            if (cr_mig_tgt)
                return (0);
            /*else if( cr_mig_src ){
               close(MPISPAWN_PARENT_FD); 
               return 0;
               } */
            else
                continue;
        }
#if 1
        else if (cleanup == 1 && ready > 1) {

            int cleanup1 = 0;
            int cleanup2 = 0;
            static int cleanup3 = 0;
            CR_DBG("***   mtpmi_process(v) cleanup ready=%d\n", ready);
            if (MPISPAWN_HAS_PARENT && FD_ISSET(MPISPAWN_PARENT_FD, &child_socks)) {
                cleanup1 = 1;
                ready--;
                read_size(MPISPAWN_PARENT_FD, &hdr, msg_hdr_s);
                //read_size (MPISPAWN_PARENT_FD, buf, hdr.msg_len);
                dbg("*****   mtpmi_process(v) 1: cleanup1 buf=%s ready=%d\n\n", buf, ready);
                rv = handle_mt_peer(MPISPAWN_PARENT_FD, &hdr);
                if (rv != 0) {
//                     PRINT_ERROR("handle_mt_peer() failed on file descriptor %d with rank %d\n", MPISPAWN_PARENT_FD, hdr.msg_rank);
                }
            }

            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                if (FD_ISSET(MPISPAWN_CHILD_FDS[i], &child_socks)) {
                    cleanup2 = 1;
                    ready--;
                    read_size(MPISPAWN_CHILD_FDS[i], &hdr, msg_hdr_s);
                    //read_size (MPISPAWN_CHILD_FDS[i], buf, hdr.msg_len);
                    dbg("**** mtpmi_process(v) : cleanup2 buf=%s ready=%d\n\n", buf, ready);
                    rv = handle_mt_peer(MPISPAWN_CHILD_FDS[i], &hdr);
                    if (rv != 0) {
//                     PRINT_ERROR("handle_mt_peer() failed on file descriptor %d with rank %d\n", MPISPAWN_CHILD_FDS[i], hdr.msg_rank);
                    }
                }
            }

            for (i = 0; 0 == rv && ready > 0 && i < NCHILD; i++) {
                if (FD_ISSET(children[i].fd, &child_socks)) {
                    ready--;
                    cleanup3 = 1;
                    dbg("***** mtpmi_process(v): cleanup3 buf=%s ready=%d\n\n", buf, ready);
                    cleanup = 0;
                    //goto handle_child;
                }
            }

            dbg("cleanup1=%d, cleanup2=%d, cleanup3=%d\n", cleanup1, cleanup2, cleanup3);
            if (!cleanup1 && !cleanup2 && cleanup3) {
                cleanup = 0;
            }
            continue;
        }
#endif
        cleanup = 0;

#else

        ready = select(FD_SETSIZE, &child_socks, NULL, NULL, NULL);
#endif

        if (ready < 0) {
            perror("select");
            mpispawn_abort(ERR_DEF);
        } else {
            CR_DBG("mtpmi_process(v) ready=%d\n", ready);
        }
        if (MPISPAWN_HAS_PARENT && FD_ISSET(MPISPAWN_PARENT_FD, &child_socks)) {
            ready--;
            read_size(MPISPAWN_PARENT_FD, &hdr, msg_hdr_s);
            dbg(">mtpmi_process(v) handle_mt_PARENT:ready=%d\n\n", ready);
            rv = handle_mt_peer(MPISPAWN_PARENT_FD, &hdr);
            if (rv != 0) {
//             PRINT_ERROR("handle_mt_peer() failed on file descriptor %d with rank %d\n", MPISPAWN_PARENT_FD, hdr.msg_rank);
            }
        }
        for (i = 0; rv == 0 && ready > 0 && i < MPISPAWN_NCHILD; i++) {
            if (FD_ISSET(MPISPAWN_CHILD_FDS[i], &child_socks)) {
                ready--;
                read_size(MPISPAWN_CHILD_FDS[i], &hdr, msg_hdr_s);
                dbg(">mtpmi_process(v) MT_CHILD (%d of %d): msg-len=%d, ready=%d\n\n", i, MPISPAWN_NCHILD, hdr.msg_len, ready);
                rv = handle_mt_peer(MPISPAWN_CHILD_FDS[i], &hdr);
                if (rv != 0) {
//             PRINT_ERROR("handle_mt_peer() failed on file descriptor %d with rank %d\n", MPISPAWN_CHILD_FDS[i], hdr.msg_rank);
                }
            }
        }
        for (i = 0; 0 == rv && ready > 0 && i < NCHILD; i++) {
            if (FD_ISSET(children[i].fd, &child_socks)) {
                ready--;
                int size = readline(children[i].fd, buf, MAXLINE);
                if ( size <= 0) {
#if defined(CKPT) && defined(CR_FTB)
                    if ( cr_mig_src ) {
                        // Do not print error message if I am a migration source
                        // Set -1 to exit mtpmi_processops()
                        rv = -1;
                        break;
                    } else {
#endif
                        PRINT_ERROR("Error while reading PMI socket. MPI process died?\n");
                        mpispawn_abort(MPISPAWN_PMI_READ_ERROR);
#if defined(CKPT) && defined(CR_FTB)
                    }
#endif
                }
                rv = parse_str(children[i].rank, children[i].fd, buf, strlen(buf), MT_RANK);
                if ( rv < 0) {
                    PRINT_ERROR("parse_str() failed with error code %d\n", rv);
                    mpispawn_abort(MPISPAWN_INTERNAL_ERROR);
                }
            }
        }
        //CR_DBG(">mtpmi_process(v) while-:ready=%d\n",ready);
    }
    dbg("%s: ret.. rv=%d\n", __func__, rv);
    return 0;
}

/* send the spawn request fully to the mpirun on the root node.
   if the spawn succeeds the mpirun will send a status. Form a response and
   send back to the waiting PMIU_readline */

char *handle_spawn_request(int fd, char *buf, int buflen)
{
    int sock;
    int event = MPISPAWN_DPM_REQ, id = env2int("MPISPAWN_ID");
    FILE *fp;
    struct stat statb;

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        /* Oops! */
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in sockaddr;
    struct hostent *mpirun_hostent;
    int spcnt, j, size;
    uint32_t totsp, retval;
    char *fname;
    mpirun_hostent = gethostbyname(env2str("MPISPAWN_MPIRUN_HOST"));
    if (NULL == mpirun_hostent) {
        /* Oops! */
        herror("gethostbyname");
        exit(EXIT_FAILURE);
    }

    sockaddr.sin_family = AF_INET;
    sockaddr.sin_addr = *(struct in_addr *) (*mpirun_hostent->h_addr_list);
    sockaddr.sin_port = htons(env2int("MPISPAWN_CHECKIN_PORT"));

    while (connect(sock, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0) ;
    if (!sock) {
        perror("connect");
        exit(EXIT_FAILURE);
    }
    /* now mpirun_rsh is waiting for the spawn request to be sent,
       read MAXLINE at a time and pump it to mpirun. */
    read_size(fd, &spcnt, sizeof(uint32_t));
    /* read in spawn cnt */

/*    fprintf(stderr, "spawn count = %d\n", spcnt); */
    read_size(fd, &totsp, sizeof(uint32_t));

/*    fprintf(stderr, "total spawn datasets = %d\n", totsp); */

    sprintf(buf, "/tmp/tempfile_socket_dump.%d", getpid());
    fname = strdup(buf);
    fp = fopen(fname, "w");
    for (j = 0; j < spcnt; j++) {
        /* Continue to readline until it receives the end of the command indicated by endcmd.
         * It first read the size of the message and then the message.
         * It writes each message in the tmp file. */
        do {
            read_size(fd, &size, sizeof(int));
            readline(fd, buf, size);
            write(fileno(fp), buf, size);
        } while (strstr(buf, "endcmd") == NULL);
    }
    fclose(fp);
    /* now we're connected to mpirun on root node */
    write(sock, &event, sizeof(int));
    write(sock, &id, sizeof(int));
    fp = fopen(fname, "r");
    fstat(fileno(fp), &statb);
    retval = statb.st_size;
    write(sock, &totsp, sizeof(uint32_t));
    write(sock, &retval, sizeof(uint32_t));

    sendfile(sock, fileno(fp), 0, retval);
    fclose(fp);
    unlink(fname);
    return NULL;
}
