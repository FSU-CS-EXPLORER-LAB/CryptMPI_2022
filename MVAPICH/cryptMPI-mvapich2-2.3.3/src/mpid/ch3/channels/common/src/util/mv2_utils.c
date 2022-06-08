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

/****Include Files****/
#include "mv2_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <ctype.h>
#include <upmi.h>

/****Function Definitions****/
/**
 * This function takes care of 'K', 'M' and 'G' present in user parameters
 * and returns equivalent binary value
 */
int user_val_to_bytes(char* value, const char* param)
{

    int factor=1;
    int length=0;
    int rank=-1;
    char lastChar='\0';
    char *str;

    UPMI_GET_RANK(&rank);
    length=strlen(value);
    lastChar=value[length-1];
    str=alloca(length);

    if (isalpha(lastChar)) {

        strncpy(str,value,length-1);
        switch(lastChar) {

            case 'k':
            case 'K':
                factor = 1<<10;
                break;
            case 'm':
            case 'M':
                factor = 1<<20;
                break;
            case 'g':
            case 'G':
                factor = 1<<30;
                break;
            default:

                if (rank==0) {

                    fprintf(stderr,"\nIllegal value in %s environment variable!\n"
                                   "Argument should be numeric. \n"
                                   "Suffixes such as 'K' (for kilobyte), 'M' (for megabyte) "
                                   "and 'G' (for gigabyte) can be used.\n\n", param);
                }

        }
    } else {

        str=value;
        factor = 1;
    }

    if (atoi(str) < 0) {
        fprintf(stderr,"\nIllegal value in %s environment variable!\n", param);
        return 4194304;
    }

    return atoi(str) * factor;
}


