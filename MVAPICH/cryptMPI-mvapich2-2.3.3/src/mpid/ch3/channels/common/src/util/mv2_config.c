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

#ifndef MV2_SYSTEM_CONFIG
#define MV2_SYSTEM_CONFIG "/etc/mvapich2.conf"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <assert.h>
#include <mv2_config.h>
#include <crc32h.h>

static struct error_info {
    char filename[FILENAME_MAX];
    unsigned lineno;
    char const * msg;
} config_error;

/******************************************************************************
 *                         Functions used internally                          *
 ******************************************************************************/

/*
 * Trim whitespace from the beginning and end of a string
 *
 * Return Value:
 *  pointer to trimmed string
 */
static char * trim_ws (char * const str)
{
    char * ptr = str;
    size_t len;

    if (NULL == ptr) {
        return NULL;
    }

    while (isspace((int)*ptr)) {
        ptr++;
    }

    len = strlen(ptr);

    while (len && isspace((int)ptr[len - 1])) {
        len--;
    }

    ptr[len] = '\0';

    return ptr;
}

/*
 * Process line to get key/value pairs
 *
 * Returns Value
 *  0   Success
 * -1   Failure
 */
static int process_line (char * const lineptr)
{
    char const * key, * value;
    char * equals, * line;

    /*
     * Check for valid input parameters
     */
    assert(lineptr != NULL);

    line = trim_ws(lineptr);
    equals = strchr(line, '=');

    /*
     * Ignore empty lines and comments
     */
    if ('\0' == line[0] || '#' == line[0]) {
        return 0;
    }

    /*
     * Could not find =
     */
    if (NULL == equals) {
        config_error.msg = "Line not of form `PARAMETER = VALUE'";
        return -1;
    }

    equals[0] = '\0';
    key = trim_ws(line);
    value = trim_ws(equals + 1);

    /*
     * Key is empty
     */
    if ('\0' == key[0]) {
        config_error.msg = "Line not of form `PARAMETER = VALUE'";
        return -1;
    }

    /*
     * Value is empty
     */
    if ('\0' == value[0]) {
        config_error.msg = "Line not of form `PARAMETER = VALUE'";
        return -1;
    }

    if (setenv(key, value, 0)) {
        config_error.msg = "Error seting environment variable";
        return -1;
    }

    return 0;
}

/*
 * Read configuration file line by line
 *
 * 1. Update CRC
 * 2. Retrieve Key/Value pairs
 */
int read_config (FILE * stream, unsigned long * crc)
{
    char lineptr[BUFSIZ];

    /*
     * Check for valid input parameters
     */
    assert(crc != NULL);
    assert(stream != NULL);
    assert(!(feof(stream) || ferror(stream)));

    while (fgets(lineptr, BUFSIZ, stream)) {
        config_error.lineno++;

        /*
         * Make sure that the line isn't longer than BUFSIZ
         */
        if (!(strchr(lineptr, '\n') || feof(stream))) {
            config_error.msg = "Line too long";
            return -1;
        }

        /*
         * Update CRC
         */
        *crc = update_crc(*crc, lineptr, strlen(lineptr));

        if (process_line(lineptr)) {
            return -1;
        }
    }

    if (feof(stream)) {
        return 0;
    }

    config_error.msg = "Error reading line";

    return -1;
}

/*
 * Read user configuration file
 */
int read_user_config (unsigned long * crc)
{
    char const * user_config = getenv("MV2_USER_CONFIG");
    char user_default[FILENAME_MAX];
    char const * ignore = getenv("MV2_IGNORE_USER_CONFIG");
    FILE * config_file;
    int report_fopen_error = 0;

    /*
     * Check for valid input parameters
     */
    assert(crc != NULL);

    /*
     * Do not read user config if MV2_IGNORE_USER_CONFIG is set.
     */
    if (ignore ? atoi(ignore) : 0) {
        return 0;
    }

    /*
     * Check if MV2_USER_CONFIG is set.  If not, use default user configuration
     * file.
     */
    if (user_config) {
        report_fopen_error = 1;
    } else if(getenv("HOME")) {
        strcpy(user_default, getenv("HOME"));
        strcat(user_default, "/.mvapich2.conf");
        user_config = user_default;
    }

    /*
     * Only read file if user_config is set.
     */
    if (user_config != NULL) {
        /*
         * Initialize error information in case of error processing file
         */
        strcpy(config_error.filename, user_config);
        config_error.lineno = 0;

        /*
         * Open and process configuration file
         */
        if ((config_file = fopen(user_config, "r"))) {
            return read_config(config_file, crc);
        } else if (report_fopen_error) {
            config_error.msg = strerror(errno);
            return -1;
        }
    }

    return 0;
}

/*
 * Read system configuration file
 */
int read_system_config (unsigned long * crc)
{
    char const * ignore = getenv("MV2_IGNORE_SYSTEM_CONFIG");
    FILE * config_file;

    /*
     * Check for valid input parameters
     */
    assert(crc != NULL);

    /*
     * Initialize error information in case of error processing file
     */
    strcpy(config_error.filename, MV2_SYSTEM_CONFIG);
    config_error.lineno = 0;

    /*
     * Do not read system config if MV2_IGNORE_SYSTEM_CONFIG is set.
     */
    if (ignore ? atoi(ignore) : 0) {
        return 0;
    }

    /*
     * Open and process configuration file
     */
    if ((config_file = fopen(MV2_SYSTEM_CONFIG, "r"))) {
        return read_config(config_file, crc);
    }

    return 0;
}

/******************************************************************************
 *                             External Functions                             *
 ******************************************************************************/

/*
 * Find and process config files
 *
 * Fills mv2_config structure with array of key/value pairs.  Each key only
 * appears once and contains the most recently seen value.
 *
 * Return value:
 *  0 on success
 *  -1 on failure
 */
extern int read_configuration_files (unsigned long * crc)
{
    /*
     * Check for valid input parameters
     */
    assert(crc != NULL);

    /*
     * Initiallize CRC
     */
    *crc = 0; gen_crc_table();

    /*
     * Read user configuration file
     */
    if (read_user_config(crc)) {
        fprintf(stderr, "[%s:%d]: %s\n", config_error.filename,
                config_error.lineno, config_error.msg);
        return -1;
    }

    /*
     * Read system configuration file
     */
    if (read_system_config(crc)) {
        fprintf(stderr, "[%s:%d]: %s\n", config_error.filename,
                config_error.lineno, config_error.msg);
        return -1;
    }

    return 0;
}

