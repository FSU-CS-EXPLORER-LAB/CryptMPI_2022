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
 * MPISPAWN INTERFACE FOR BUILDING DYNAMIC SOCKET TREES
 */

#include "mpispawn_tree.h"
#include "mpirun_util.h"
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "mpichconf.h"
#include <math.h>
#include <netdb.h>

#ifdef MPISPAWN_DEBUG
#include <stdio.h>
#define debug(...) fprintf(stderr, __VA_ARGS__)
#else
#define debug(...) ((void)0)
#endif

//#define dbg(fmt, args...)   printf("%s: "fmt, __func__, ##args)
#define dbg(fmt, args...)

extern int MPISPAWN_HAS_PARENT;
extern int MPISPAWN_NCHILD;

typedef struct {
    size_t num_parents, num_children;
} family_size;

typedef struct {
    char node[MAX_HOST_LEN + 1];
    char serv[MAX_PORT_LEN + 1];
} mpispawn_info_t;

#if defined(CKPT) && defined(CR_FTB)
/*struct spawn_info_s {
    char spawnhost[32];
    int  sparenode;
}; */

extern struct spawn_info_s *spawninfo;
extern int exclude_spare;
#endif

static size_t id;
static int l_socket;
static mpispawn_info_t *mpispawn_info;

typedef enum {
    CONN_SUCCESS,
    CONN_LIB_FAILURE,
    CONN_VERIFY_FAILURE
} CONN_STATUS;

#if 0                           // unused (will be remove later)
static CONN_STATUS conn2parent(size_t parent, int *p_socket)
{
    size_t p_id;

    debug("entering conn2parent [id: %d]\n", id);
    while ((*p_socket = accept(l_socket, (struct sockaddr *) NULL, NULL)) < 0) {
        switch (errno) {
        case EINTR:
        case EAGAIN:
            continue;
        default:
            return CONN_LIB_FAILURE;
        }
    }

    debug("verifying conn2parent [id: %d]\n", id);
    /*
     * Replace the following with a simple check of the sockaddr filled in
     * by the accept call with the sockaddr stored at mpispawn_info[arg->id]
     */
    if (read_socket(*p_socket, &p_id, sizeof(p_id))
        || p_id != parent || write_socket(*p_socket, &id, sizeof(id))) {

        close(*p_socket);
        return CONN_VERIFY_FAILURE;
    }

    debug("leaving conn2parent [id: %d]\n", id);
    return CONN_SUCCESS;
}

static CONN_STATUS conn2children(size_t const n, size_t const children[], int c_socket[])
{
    size_t i, c_id;

    for (i = 0; i < n; ++i) {
        c_socket[i] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        if (c_socket[i] < 0) {
            while (i)
                close(c_socket[--i]);
            return CONN_LIB_FAILURE;
        }

        if (connect(c_socket[i], (struct sockaddr *) &mpispawn_info[children[i]], sizeof(struct sockaddr)) < 0) {
            while (i)
                close(c_socket[--i]);
            return CONN_LIB_FAILURE;
        }

        if (write_socket(c_socket[i], &id, sizeof(id))
            || read_socket(c_socket[i], &c_id, sizeof(c_id))
            || c_id != children[i]) {
            while (i)
                close(c_socket[--i]);
            return CONN_VERIFY_FAILURE;
        }
    }

    return CONN_SUCCESS;
}
#endif

static family_size find_family(size_t const root, size_t const degree, size_t const node_count, size_t * parent, size_t children[])
{
    size_t offset = node_count - root;
    size_t position = (id + offset) % node_count;
    size_t c_start = degree * position + 1;
    size_t i;
    family_size fs = { 0, 0 };

    /*
     * Can't build a tree when nodes have a degree of 0
     */
    if (!degree) {
        return fs;
    }

    else if (position) {
        *parent = ((position - 1) / degree + root) % node_count;
        fs.num_parents = 1;
    }

    /*
     * Find the number of children I have
     */
    if (c_start < node_count) {
        if (degree > node_count - c_start) {
            i = fs.num_children = node_count - c_start;
        }

        else {
            i = fs.num_children = degree;
        }

        while (i--) {
            children[i] = (c_start + i + root) % node_count;
        }
    }

    return fs;
}

extern int *mpispawn_tree_init(size_t me, const size_t degree, const size_t node_count, int req_socket)
{
    size_t parent, child[degree];
    size_t i;
    int p_socket;
    family_size fs;
    int *socket_array;
    extern size_t id;
    extern int l_socket;

    id = me;
    l_socket = req_socket;

    fs = find_family(0, degree, node_count, &parent, child);
    MPISPAWN_NCHILD = fs.num_children;
    MPISPAWN_HAS_PARENT = fs.num_parents;

    socket_array = (int *) calloc(fs.num_parents + fs.num_children, sizeof(int));

    if (!socket_array) {
        perror("calloc");
        return NULL;
    }

    memset(socket_array, 0xff, (fs.num_parents + fs.num_children) * sizeof(int));

    /*
     * Connect to parent
     */
    debug("[id: %d] connecting to parent\n", id);

    int flags, nonblocking = 1;
    fd_set set;
    struct timeval tv;

    flags = fcntl(l_socket, F_GETFL);
    if (flags < 0) {
        nonblocking = 0;
    } else {
        if (fcntl(l_socket, F_SETFL, flags | O_NONBLOCK))
            nonblocking = 0;
    }

    if (nonblocking) {
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        FD_ZERO(&set);
        FD_SET(l_socket, &set);
        if (select(l_socket + 1, &set, NULL, NULL, &tv) < 0) {
            goto free_socket_array;
        }
    }

    while ((p_socket = socket_array[0] = accept(l_socket, (struct sockaddr *) NULL, NULL)) < 0) {
        switch (errno) {
        case EINTR:
        case EAGAIN:
            continue;
        default:
            perror("mpispawn_tree_init");
            goto free_socket_array;
        }
    }

    debug("[id: %d] connected to parent\n", id);

    mpispawn_info = (mpispawn_info_t *) calloc(sizeof(mpispawn_info_t),
            node_count);
    if (!mpispawn_info) {
        perror("mpispawn_tree_init");
        goto close_p_socket;
    }

    if (read_socket(p_socket, mpispawn_info, sizeof(mpispawn_info_t) * node_count)) {
        perror("mpispawn_tree_init");
        goto free_mpispawn_info;
    }
#if defined(CKPT) && defined(CR_FTB)
    spawninfo = (struct spawn_info_s *) calloc(sizeof(struct spawn_info_s), node_count);
    if (!spawninfo) {
        perror("[CR_MIG] calloc(spawninfo)");
        goto free_mpispawn_info;
    }

    if (read_socket(p_socket, spawninfo, sizeof(struct spawn_info_s) * node_count)) {
        perror("[CR_MIG] read_socket(spawninfo)");
        goto free_spawninfo;
    }

    for (i = 0; i < node_count; i++) {
        debug("***** %s:%d[mpispawn:spawninfo:%d] %s - %d\n", __FILE__, __LINE__, i, spawninfo[i].spawnhost, spawninfo[i].sparenode);
    }
#endif

    for (i = 0; i < fs.num_children; ++i) {
        int c_socket = connect_socket(mpispawn_info[child[i]].node,
                mpispawn_info[child[i]].serv);

        if (0 == c_socket) {
            goto free_spawninfo;
        }

        socket_array[fs.num_parents + i] = c_socket;

        if (write_socket(c_socket, mpispawn_info, sizeof(mpispawn_info_t) * node_count)) {
            do {
                close(socket_array[fs.num_parents + i]);
            } while (i--);

            goto free_spawninfo;
        }
#if defined(CKPT) && defined(CR_FTB)
        if (write_socket(c_socket, spawninfo, sizeof(struct spawn_info_s) * node_count)) {
            do {
                close(socket_array[fs.num_parents + i]);
            } while (i--);

            goto free_spawninfo;
        }
#endif
    }

    /*
     * close connection to mpirun_rsh
     */
    if (id == 0) {
        close(p_socket);
    }
#if defined(CKPT) && defined(CR_FTB)
    int index_spawninfo = 0;
    static char my_host_name[MAX_HOST_LEN];
    gethostname(my_host_name, MAX_HOST_LEN);
    debug("===== id= %d -- num_parents %d -- NUMCHILDREN %d %s \n", id, fs.num_parents, fs.num_children, my_host_name);
    debug("id= %d -- MPISPAWN_HAS_PARENT %d\n", id, MPISPAWN_HAS_PARENT);
    debug("id= %d -- PARENT %d\n", id, parent);
    debug("id= %d -- DEGREE %d\n", id, degree);
    debug("id= %d -- NUMCHILDREN %d\n", id, fs.num_children);
    for (i = 0; i < fs.num_children; i++) {
        index_spawninfo = (((MPISPAWN_HAS_PARENT) ? id : 0) * degree) + i;
        debug("%s:%d[mpispawn_tree_connect:%d] %d Child(%d) =%d\n", __FILE__, __LINE__, ((MPISPAWN_HAS_PARENT) ? parent : 0), id, i, index_spawninfo);

        if (spawninfo[index_spawninfo + 1].sparenode) {
            ++exclude_spare;
        }
    }

    debug("[%d on %s] exclude_spare::%d\n", id, my_host_name, exclude_spare);
    debug("%s:%d:mpispawn_id::%d exclude_spare::%d\n", __FILE__, __LINE__, id, exclude_spare);
#endif

    debug("leaving mpispawn_tree_init [id: %d]\n", me);
    return socket_array;

  free_spawninfo:
#if defined(CKPT) && defined(CR_FTB)
    free(spawninfo);
#endif

  free_mpispawn_info:
    free(mpispawn_info);

  close_p_socket:
    close(p_socket);

  free_socket_array:
    free(socket_array);

    return NULL;
}
