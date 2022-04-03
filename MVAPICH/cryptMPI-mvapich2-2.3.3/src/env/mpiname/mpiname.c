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

#include "mpi.h"
#include <unistd.h>
#include <stdio.h>

extern const char MPIR_Version_string[];
extern const char MPIR_Version_date[];
extern const char MPIR_Version_configure[];
extern const char MPIR_Version_device[];
extern const char MPIR_Version_CC[];
extern const char MPIR_Version_CXX[];
extern const char MPIR_Version_F77[];
extern const char MPIR_Version_FC[];
#define MPINAME_VERSION MPIR_Version_string
#define MPINAME_RELEASE_DATE MPIR_Version_date
#define MPINAME_OPTIONS MPIR_Version_configure
#define MPINAME_DEVICE MPIR_Version_device
#define MPINAME_CC MPIR_Version_CC
#define MPINAME_CXX MPIR_Version_CXX
#define MPINAME_F77 MPIR_Version_F77
#define MPINAME_FC MPIR_Version_FC
#define MPINAME_NAME "MVAPICH2"

#define PRINT_DEVICE 8
#define PRINT_NAME 1
#define PRINT_RELEASE_DATE 4
#define PRINT_VERSION 2
static unsigned char s_lineElements = 0;

#define PRINT_COMPILERS 1
#define PRINT_OPTIONS 2
static unsigned char s_stackElements = 0;

void print_element (unsigned int mask, const char element[])
{
    if (s_lineElements & mask)
    {
        s_lineElements ^= mask;
        printf("%s%s", element, s_lineElements ? " " : s_stackElements ? "\n\n" : "\n");
    }
}

void usage ()
{
    printf("Usage: [OPTION]...\n");
    printf("Print MPI library information.  With no OPTION, the output is the same as -v.\n\n");
    printf("  -a    print all information\n");
    printf("  -c    print compilers\n");
    printf("  -d    print device\n");
    printf("  -h    display this help and exit\n");
    printf("  -n    print the MPI name\n");
    printf("  -o    print configuration options\n");
    printf("  -r    print release date\n");
    printf("  -v    print library version\n\n");
}

int main (int argc, char* argv[])
{
    int c;

    while ((c = getopt(argc, argv, "acdhnorv")) != -1)
    {
        switch (c)
        {
        case 'a':
                s_lineElements = PRINT_DEVICE
                    | PRINT_NAME
                    | PRINT_RELEASE_DATE
                    | PRINT_VERSION;
                s_stackElements = PRINT_COMPILERS | PRINT_OPTIONS;
            break;
        case 'c':
                s_stackElements = PRINT_COMPILERS;
            break;
        case 'd':
                s_lineElements |= PRINT_DEVICE;
            break;
        case 'h':
                usage();
            return 1;
        case 'n':
                s_lineElements |= PRINT_NAME;
            break;
        case 'o':
                s_stackElements |= PRINT_OPTIONS;
            break;
        case 'r':
                s_lineElements |= PRINT_RELEASE_DATE;
            break;
        case 'v':
                s_lineElements |= PRINT_VERSION;
            break;
        default: 
                usage();
            return 1;
        }
    }

    if (!s_lineElements && !s_stackElements)
    {
        s_lineElements = PRINT_NAME;
    }

    print_element(PRINT_NAME, MPINAME_NAME);
    print_element(PRINT_VERSION, MPINAME_VERSION);
    print_element(PRINT_RELEASE_DATE, MPINAME_RELEASE_DATE);
    print_element(PRINT_DEVICE, MPINAME_DEVICE);
 
    if (s_stackElements & PRINT_COMPILERS)
    {
        s_stackElements ^= PRINT_COMPILERS;
        printf("Compilation\nCC: %s\nCXX: %s\nF77: %s\nFC: %s\n\n",
            MPINAME_CC,
            MPINAME_CXX,
            MPINAME_F77,
            MPINAME_FC);
    }
    
    if (s_stackElements & PRINT_OPTIONS)
    {
        printf("Configuration\n%s\n\n", MPINAME_OPTIONS);
    }

    return 0;
}

