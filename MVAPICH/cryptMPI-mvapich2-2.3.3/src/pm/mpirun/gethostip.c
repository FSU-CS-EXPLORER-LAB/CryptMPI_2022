/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#include <stdlib.h>
#include <limits.h>
#include <unistd.h>

#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

int
gethostip(char * ipstr, size_t ipstr_len)
{
    struct addrinfo hints;
    struct addrinfo *result;
    char node[HOST_NAME_MAX + 1];
    int s;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_flags = 0;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = 0;
    hints.ai_addrlen = 0;
    hints.ai_addr = NULL;
    hints.ai_canonname = NULL;
    hints.ai_next = NULL;

    gethostname(node, sizeof(node));
    node[HOST_NAME_MAX] = '\0'; /* Just in case */
    s = getaddrinfo(node, "0", &hints, &result);

    if (s) {
        return -1;
    }

    s = getnameinfo(result->ai_addr, result->ai_addrlen, ipstr, ipstr_len,
            NULL, 0, NI_NUMERICHOST | NI_NUMERICSERV);

    freeaddrinfo(result);

    if (s) {
        return -1;
    }

    return 0;
}
