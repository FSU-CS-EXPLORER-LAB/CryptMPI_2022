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
#include <signal_processor.h>

#ifdef CR_AGGRE

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <sys/wait.h>
#include <attr/xattr.h>

#include "crfs.h"
#include "debug.h"
#include "debug_utils.h"

#define    MAX_PATH_LEN    (128)    // length of mount-point
#define MAX_CMD_LEN    (MAX_PATH_LEN*2)

// defined in mpispawn.c
extern int mt_id;

char crfs_wa_real[MAX_PATH_LEN];
char crfs_wa_mnt[MAX_PATH_LEN];
int crfs_wa_pid = 0;

char crfile_basename[MAX_PATH_LEN]; // base filename of CR files

char crfs_mig_real[MAX_PATH_LEN];
char crfs_mig_mnt[MAX_PATH_LEN];
int crfs_mig_pid = 0;

char crfs_mig_filename[MAX_PATH_LEN];   // like:  /tmp/cr-<sessionid>/mig/myfile

char crfs_sessionid[MAX_PATH_LEN];

extern int crfs_mode;
extern int mig_role;

static long parse_value_string(char *msg);

int crfs_start_mig(char *tgt)
{
    int ret;
    ret = lsetxattr(crfs_mig_mnt, "migration.tgt", tgt, strlen(tgt), 0);

    int run = 1;
    ret = lsetxattr(crfs_mig_mnt, "migration.state", &run, sizeof(int), 0);

    dbg("***   have started mig to %s\n", tgt);
    return ret;
}

int crfs_stop_mig()
{
    int ret;

    int run = 0;
    ret = lsetxattr(crfs_mig_mnt, "migration.state", &run, sizeof(int), 0);

    dbg("****  have stopped mig\n");
    return ret;
}

// ckptfile is:  ${real-dir}/filename-base. 
// Need to extract "${real-dir}" and "file-basename" from it
int start_crfs_wa(char *sessionid, char *realdir)
{
    char cmd[MAX_CMD_LEN];

    memset(crfs_wa_real, 0, MAX_PATH_LEN);
    memset(crfs_wa_mnt, 0, MAX_PATH_LEN);
    memset(cmd, 0, MAX_CMD_LEN);

    strncpy(crfs_wa_real, realdir, MAX_PATH_LEN);
    snprintf(cmd, MAX_CMD_LEN, "mkdir -p %s", crfs_wa_real);
    system(cmd);

    snprintf(crfs_wa_mnt, MAX_PATH_LEN, "/tmp/cr-%s/wa/", sessionid);

    snprintf(cmd, MAX_CMD_LEN, "mkdir -p %s", crfs_wa_mnt);
    system(cmd);

    int pfd[2];
    char ch;

    int argc = 0;
    char *argv[10];
    int fg = 0;

    crfs_mode = MODE_WRITEAGGRE;
    mig_role = ROLE_INVAL;

    argv[0] = "crfs-wa";
    argv[1] = crfs_wa_real;
    argv[2] = crfs_wa_mnt;
    argv[3] = "-obig_writes";   //NULL; //"-odirect_io"; //"-obig_writes";
    argv[4] = NULL;
    argc = 4;
    if (fg) {
        argv[argc] = "-f";
        argc++;
        argv[argc] = NULL;
    }

    dbg("real-dir=%s, mnt=%s\n", argv[1], argv[2]);

    if (pipe(pfd) != 0) {
        perror("Fail to create pipe...\n");
        return -1;
    }

    crfs_wa_pid = fork();

    if (crfs_wa_pid == 0)       // in child proc
    {
        clear_sigmask();

        // Set prefix for debug output
        {
            const int MAX_LENGTH = 256;
            char hostname[MAX_LENGTH];
            gethostname(hostname, MAX_LENGTH);
            hostname[MAX_LENGTH - 1] = '\0';
            char output_prefix[MAX_LENGTH];
            snprintf(output_prefix, MAX_LENGTH, "%s:mpispawn_wa_%i", hostname, mt_id);
            set_output_prefix(output_prefix);
        }

        extern void *crfs_main(int pfd, int argc, char **argv);
        close(pfd[0]);          // close the read-end
        PRINT_DEBUG(DEBUG_Fork_verbose, "start CRFS-wa (pid=%d)\n", getpid());
        crfs_main(pfd[1], argc, argv);
        PRINT_DEBUG(DEBUG_Fork_verbose, "CRFS-wa will exit now...\n");
        exit(0);
    } else if (crfs_wa_pid < 0) // error!! 
    {
        perror("fail to fork...\n");
        return -1;
    }

    PRINT_DEBUG(DEBUG_Fork_verbose, "FORK mpispawn_wa (pid=%d)\n", crfs_wa_pid);
    /// pid>0: in parent proc
    close(pfd[1]);              // close the write-end
    dbg("parent proc waits...\n\n");
    read(pfd[0], &ch, 1);       // wait for a sig
    dbg("*****   has got a char ==  %c\n", ch);
    close(pfd[0]);
    if (ch != '0') {
        stop_crfs_wa();
        return -1;
    }
    return 0;
}

/// src_tgt::  0=at src, 1=srv tgt
int start_crfs_mig(char *sessionid, int src_tgt)
{
    char cmd[MAX_CMD_LEN];

    if (src_tgt != 1 && src_tgt != 2) {
        err("Incorrect param: src_tgt=%d\n", src_tgt);
        return -1;
    }
    //realdir = ckptfile;

    memset(crfs_mig_real, 0, MAX_PATH_LEN);
    memset(crfs_mig_mnt, 0, MAX_PATH_LEN);
    memset(cmd, 0, MAX_CMD_LEN);

    snprintf(crfs_mig_real, MAX_PATH_LEN, "/tmp/cr-%s/", sessionid);
    snprintf(cmd, MAX_CMD_LEN, "mkdir -p %s", crfs_mig_real);
    system(cmd);

    snprintf(crfs_mig_mnt, MAX_PATH_LEN, "/tmp/cr-%s/mig/", sessionid);
    snprintf(cmd, MAX_CMD_LEN, "mkdir -p %s", crfs_mig_mnt);
    system(cmd);

    int pfd[2];
    char ch;

    int argc = 0;
    char *argv[10];             //
    //{ "crfs-wa",  "/tmp/ckpt", "/tmp/mnt", "-f", "-odirect_io", NULL };

    if (src_tgt == 1)           // I'm mig-source
    {
        crfs_mode = MODE_WRITEAGGRE;
        mig_role = ROLE_MIG_SRC;
    } else if (src_tgt == 2)    // at target side
    {
        crfs_mode = MODE_MIG;
        mig_role = ROLE_MIG_TGT;
    }
    int fg = 0;

    argv[0] = "crfs-mig";
    argv[1] = crfs_mig_real;
    argv[2] = crfs_mig_mnt;
    argv[3] = "-osync_read";
    argv[4] = "-obig_writes";   //NULL; //"-odirect_io"; //"-obig_writes";
    argv[5] = NULL;
    argc = 5;
    if (fg) {
        argv[argc++] = "-f";
        argv[argc] = NULL;
    }

    if (pipe(pfd) != 0) {
        perror("Fail to create pipe...\n");
        return -1;
    }

    crfs_mig_pid = fork();

    if (crfs_mig_pid == 0)      // in child proc
    {
        clear_sigmask();

        // Set prefix for debug output
        {
            const int MAX_LENGTH = 256;
            char hostname[MAX_LENGTH];
            gethostname(hostname, MAX_LENGTH);
            hostname[MAX_LENGTH - 1] = '\0';
            char output_prefix[MAX_LENGTH];
            snprintf(output_prefix, MAX_LENGTH, "%s:mpispawn_mig_%i", hostname, mt_id);
            set_output_prefix(output_prefix);
        }

        extern void *crfs_main(int pfd, int argc, char **argv);
        close(pfd[0]);          // close the read-end
        PRINT_DEBUG(DEBUG_Fork_verbose, "start CRFS-mig (pid=%d)\n", getpid());
        crfs_main(pfd[1], argc, argv);
        PRINT_DEBUG(DEBUG_Fork_verbose, "CRFS-mig will exit now...\n");
        exit(0);
    } else if (crfs_mig_pid < 0)    // error!! 
    {
        perror("fail to fork...\n");
        return -1;
    }
    PRINT_DEBUG(DEBUG_Fork_verbose, "FORK mpispawn_mig (pid=%d)\n", crfs_mig_pid);

    /// pid>0: in parent proc
    close(pfd[1]);              // close the write-end
    dbg("parent proc waits...\n\n");
    read(pfd[0], &ch, 1);       // wait for a sig
    dbg("has got a char: %c\n", ch);
    close(pfd[0]);
    if (ch != '0') {
        stop_crfs_mig();
        return -1;
    }

    return 0;

}

// the ckptfile is: ${dir}/filename. Parse this string to 
// extract ${dir} and filename.
int parse_ckptname(char *ckptfile, char *out_dir, char *out_file)
{
    int i;

    int len = strlen(ckptfile);
    char *p;

    p = ckptfile + len - 1;
    while (len) {
        if (*p == '/')          // find the last "/", any before it is the dir-path
        {
            break;
        }
        len--;
        p--;
    }

    if (len <= 0)               // something is wrong. ill-formated input ckptfile name
    {
        err("incorrect ckpt name: %s\n", ckptfile);
        return -1;
    }

    strncpy(out_dir, ckptfile, len);
    out_dir[len] = 0;

    i = strlen(ckptfile) - len;
    strncpy(out_file, p + 1, i);
    out_file[i] = 0;

    return 0;
}

///
static int check_dir(char *dirpath)
{
    int rv;
    struct stat sbuf;

    rv = stat(dirpath, &sbuf);
    if (rv) {
        if (errno == ENOENT) {
            rv = mkdir(dirpath, 0755);
            dbg("create dir: %s ret %d\n", dirpath, rv);
            return rv;
        }
        err("Fail to open dir:  %s\n", dirpath);
        return rv;
    }

    if (!S_ISDIR(sbuf.st_mode)) {
        err("path: %s isn't a dir!!\n", dirpath);
        rv = -1;
    }

    return rv;
}

extern long cli_rdmabuf_size, srv_rdmabuf_size;
extern int rdmaslot_size;

static int has_mig_fs = 0;

int start_crfs(char *sessionid, char *fullpath, int mig)
{
    int rv;
    char realdir[256];

    if (parse_ckptname(fullpath, realdir, crfile_basename) != 0) {
        printf("%s: Error at parsing ckfile: %s\n", __func__, fullpath);
        return -1;
    }
    if (check_dir(realdir) != 0) {
        return -1;
    }
    strcpy(crfs_sessionid, sessionid);
    dbg("parse fullpath: %s to %s : %s \n", fullpath, realdir, crfile_basename);

    /// now, init the bufpool & chunk-size
    long val;
    char *p;
    p = getenv("MV2_CKPT_AGGREGATION_BUFPOOL_SIZE");
    if (p && (val = parse_value_string(p)) > 0) {
        srv_rdmabuf_size = cli_rdmabuf_size = val;
    }
    p = getenv("MV2_CKPT_AGGREGATION_CHUNK_SIZE");
    if (p && (val = parse_value_string(p)) > 0) {
        rdmaslot_size = (int) val;
    }
    dbg("cli_rdmabuf_size=%ld, srv_rdmabuf_size=%ld, slot-size=%d\n", cli_rdmabuf_size, srv_rdmabuf_size, rdmaslot_size);

    rv = start_crfs_wa(sessionid, realdir);
    dbg("[mt_%d]: Start WA ret %d\n", mt_id, rv);

    if (rv != 0) {
        err("Fail to start CR-aggregation...\n");
        return rv;
    }
    // this "fullpath" is used in aggre-based ckpt
    snprintf(fullpath, MAX_PATH_LEN, "%s%s", crfs_wa_mnt, crfile_basename);
    dbg("Now, def cktp file=%s\n", fullpath);
    dbg("---------  crfs-mig func=%d\n", mig);

    has_mig_fs = 0;
    if (mig > 0) {
        rv = start_crfs_mig(sessionid, mig);
        dbg("[mt_%d]: Start Mig ret %d\n", mt_id, rv);
        if (rv == 0) {
            has_mig_fs = mig;
            // this mig_filename is the filename used in aggre-based migration
            snprintf(crfs_mig_filename, MAX_PATH_LEN, "%s%s", crfs_mig_mnt, crfile_basename);
        } else {
            err("Fail to start Aggre-for-Migration...\n");
            stop_crfs_wa();
            return rv;
        }
    }

    return rv;
}

// string format is:   xxx<Kk/Mm/Gg>
static long parse_value_string(char *msg)
{
    int len;
    if (!msg || (len = strlen(msg)) < 1)
        return -1L;
    char c = msg[len - 1];

    unsigned long val;
    unsigned long unit;
    switch (c) {
    case 'k':
    case 'K':
        unit = 1024;
        break;
    case 'm':
    case 'M':
        unit = 1024 * 1024;
        break;
    case 'g':
    case 'G':
        unit = 1024UL * 1024 * 1024;
        break;
    default:
        unit = 1;
        break;
    }

    val = atol(msg) * unit;

    return val;

}

int stop_crfs(const char *crfs_mnt, int crfs_pid)
{
    extern int mt_id;
    int rv;
    char cmd[MAX_CMD_LEN];

    if (strlen(crfs_mnt) != 0) {
        snprintf(cmd, MAX_CMD_LEN, "fusermount -u %s > /dev/null 2>&1", crfs_mnt);
        PRINT_DEBUG(DEBUG_Fork_verbose, "Stop CRFS: %s\n", cmd);
        rv = system(cmd);
        if (rv == -1) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "system call to fusermount failed\n");
        } else {
            PRINT_DEBUG(DEBUG_Fork_verbose, "system call to fusermount returned %d\n", rv);
            if (WIFEXITED(rv)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "fusermount exited with status %d\n", WEXITSTATUS(rv));
            } else if (WIFSIGNALED(rv)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "fusermount terminated with signal %d\n", WTERMSIG(rv));
            } else if (WIFSTOPPED(rv)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "fusermount stopped with signal %d\n", WSTOPSIG(rv));
            } else if (WIFCONTINUED(rv)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "fusermount continued\n");
            }
        }
    }
    if (crfs_pid > 0) {
        rv = kill(crfs_pid, SIGTERM);
        PRINT_DEBUG(DEBUG_Fork_verbose, "kill with SIGTERM ret=%d\n", rv);
        usleep(100000);         // wait for CRFS to terminate
        rv = kill(crfs_pid, SIGINT);
        PRINT_DEBUG(DEBUG_Fork_verbose, "kill with SIGINT ret=%d\n", rv);
    }
    //snprintf(path, MAX_PATH_LEN, "/tmp/cr-%s/", crfs_sessionid);
    //rv = rmdir(path);
    return 0;
}

int stop_crfs_mig()
{
    PRINT_DEBUG(DEBUG_Fork_verbose, "Stop RDMA-migration CRFS\n");
    return stop_crfs(crfs_mig_mnt, crfs_mig_pid);
}

int stop_crfs_wa()
{
    PRINT_DEBUG(DEBUG_Fork_verbose, "Stop Write-Aggregation CRFS\n");
    return stop_crfs(crfs_wa_mnt, crfs_wa_pid);
}

#endif
