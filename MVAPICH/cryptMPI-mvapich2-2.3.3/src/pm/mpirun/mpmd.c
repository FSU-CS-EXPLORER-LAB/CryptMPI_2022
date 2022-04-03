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

#include <mpmd.h>
#include <src/pm/mpirun/mpirun_rsh.h>

#include <ctype.h>

//These are used when activated mpmd

int configfile_on = 0;
char configfile[CONFILE_LEN + 1];

/*
 * Try to kill stdio, but fail quietly if unsuccessful.
 */

/*
 * Error, fatal.
 */
void error(const char *fmt, ...)
{
    va_list ap;

    fprintf(stderr, "Error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, ".\n");

    exit(1);
}

/**
 * Push the information about the executable in the list.
 */
void push(config_spec_t ** headRef, char *exe, char *args, int numprocs)
{

    config_spec_t *newNode = (config_spec_t *) malloc(sizeof(config_spec_t));
    newNode->exe = strdup(exe);
    if (args != NULL) {
        newNode->argc = count_args(args);
        newNode->args = args;
    } else {
        newNode->args = NULL;
        newNode->argc = 1;
    }
    newNode->numprocs = numprocs;
    newNode->next = *headRef;   // The '*' to dereferences back to the real head
    *headRef = newNode;

}

/*
 * This function is in part equal to the one present in mpiexec project from OSC.
 * Read the heterogenous config file, making sure it's proper and all the
 * executables exist.  Command-line node limits have already been applied
 * and the tasks[] list reduced accordingly, except for -numproc.
 */
process *parse_config(char *configfile, int *nprocs)
{
    FILE *fp;
    char buf[16384];
    char *exe = NULL;
    char *args = NULL;

    int numprocs = 0;
    int line;

    config_spec_t *head = NULL;
    config_spec_t **lastPtrRef = &head;

    line = 0;
    if (!strcmp(configfile, "-"))
        fp = stdin;
    else {
        if (!(fp = fopen(configfile, "r"))) {
            fprintf(stderr, "Error in open \"%s\" \n", configfile);
            exit(1);
        }
    }
    //Read the configuration file
    while (fgets(buf, sizeof(buf), fp)) {
        char *cp;
        ++line;
        if (strlen(buf) == sizeof(buf) - 1) {
            error("%s: line %d too long", __func__, line);
        }

        /*
         * These isspace() casts avoid a warning about
         * "subscript has type char" on old gcc on suns.
         */
        for (cp = buf; *cp && isspace((int) *cp); cp++) {
            //printf("cp %s  \n",cp);
        }

        if (*cp == '#' || !*cp)
            continue;           /* comment or eol */

        /* run up and find the executable (after the ':') and save it */

        {
            char *cq, *cr, c;
            for (cq = cp; *cq && *cq != '#' && *cq != ':'; cq++) {  /*printf("* cp %s cq %c \n",cp,*cq); */
            }
            if (*cq != ':')
                error("%s: line %d: no ':' separating executable", __func__, line);

            *cq = 0;            /* colon -> 0, further parsing easier */
            for (++cq; *cq && isspace((int) *cq); cq++) ;
            //{printf("** cp %s cq %c \n",cp,cq);}

            if (!*cq || *cq == '#')
                error("%s: line %d: no executable after the ':'", __func__, line);

            for (cr = cq + 1; *cr && *cr != '#'; cr++) ;

            if (*cr == '#')     /* delete trailing comment */
                *cr = 0;

            for (--cr; cr > cq && isspace((int) *cr); cr--)
                *cr = '\0';     /* delete trailing space */

            for (cr = cq + 1; *cr && !isspace((int) *cr); cr++) ;

            c = *cr;
            *cr = 0;
            exe = (char *) strdup(cq);
            *(cq = cr) = c;
            for (; *cq && isspace((int) *cq); cq++) ;

            if (*cq) {
                /* Fill the list of arguments. */
                args = (char *) strdup(cq);

            }

        }

        /*
         * One possible left hand sides:
         *   -n <numproc> : exe1
         */
        if (*cp == '-') {
            if (*++cp == 'n') {

                long l;
                char *cq;
                for (++cp; *cp && isspace((int) *cp); cp++) ;

                l = strtol(cp, &cq, 10);
                //printf("l %d \n",l);
                if (l <= 0)
                    error("%s: line %d: \"-n <num>\" must be positive integer", __func__, line);
                for (cp = cq; *cp && isspace((int) *cp); cp++) ;
                if (*cp)
                    error("%s: line %d: junk after \"-n <num>\"", __func__, line);

                numprocs = l;
                *nprocs = *nprocs + l;

            } else
                error("%s: line %d: unknown \"-\" argument", __func__, line);
        }

        push(lastPtrRef, exe, args, numprocs);  // Add node at the last pointer in the list
        lastPtrRef = &((*lastPtrRef)->next);

    }
    if (fp != stdin)
        fclose(fp);

    /*TODO Some check if the -np is specified and if the list of executables is empty */

    return save_plist(head, *nprocs);
}

/**
 * Save the information read in the config file in the plist used by mpirun_rsh.
 */
process *save_plist(config_spec_t * cfg_list, int nprocs)
{
    /*
     * Initialize the plist
     */
    process *plist = malloc((nprocs) * sizeof(process));

    if (plist == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    /* Now save the information read in the config file in the plist. */
    int p = 0;
    while (cfg_list) {
        int c = 0;
        while (c < cfg_list->numprocs) {
            plist[p].state = P_NOTSTARTED;
            plist[p].device = NULL;
            plist[p].port = -1;
            plist[p].remote_pid = 0;
            plist[p].executable_name = (char *) strdup(cfg_list->exe);
            if (cfg_list->args != NULL) {
                plist[p].executable_args = (char *) strdup(cfg_list->args);
                plist[p].argc = cfg_list->argc;
            } else {
                plist[p].executable_args = NULL;
                plist[p].argc = 1;
            }
            c++;
            p++;
        }
        //printf("ssssss==== %s %s %d \n", cfg_list->exe, cfg_list->args, cfg_list->numprocs);
        cfg_list = cfg_list->next;

    }

    /* int i;
       for (i = 0; i < nprocs; i++) {
       printf("===============\n %s %s %d \n",plist[i].executable_name, plist[i].executable_args,plist[i].remote_pid);
       } */
    return plist;
}

/**
 * This method add the name of the executable and its arguments to mpispawn_env.
 *
 */
char *add_argv(char *mpispawn_env, char *exe, char *args, int tmp_i)
{

    char *cp;
    char *tmp = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env, tmp_i++, exe);
    //free(mpispawn_env);
    mpispawn_env = tmp;
    /*The args are as a single string, instead in mpispawn we need to pass each word as a single argument. */
    if (args != NULL) {
        //Add the args of the executable
        for (cp = args; *cp;) {

            char *cq, c;
            /* select a word */
            for (cq = cp + 1; *cq && !isspace((int) *cq); cq++) ;
            c = *cq;
            *cq = 0;
            /*Add each word as argument of the executable in mpispawn. */
            tmp = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env, tmp_i++, cp);
            //free(mpispawn_env);
            mpispawn_env = tmp;
            *cq = c;            /* put back delimiter */
            cp = cq;            /* advance to next word, and skip space */
            for (; *cp && isspace((int) *cp); cp++) ;
        }
    }

    return tmp;
}

/**
 * Count the number of argument of a single exe.
 *
 */
int count_args(char *args)
{
    char *cp;
    int argc = 1;
    /*The args are as a single string, we need a list of argument. */
    if (args != NULL) {
        //Add the args of the executable
        for (cp = args; *cp;) {

            char *cq, c;
            /* select a word */
            for (cq = cp + 1; *cq && !isspace((int) *cq); cq++) ;
            c = *cq;
            *cq = 0;
            argc++;
            *cq = c;            /* put back delimiter */
            /* advance to next word, and skip space */
            for (cp = cq; *cp && isspace((int) *cp); cp++) ;
        }
    }
    return argc;
}

/**
 * Insert in the host_list the name of the executable and the arguments.
 * When mpmd is activated in spawn_one we send the list of hosts and for each host we send the
 * executable name, the number of argument and the list of argument.
 * The host_list has the following form: host1:numProc:pid1:pid2..:pidN:exe:argc:arg1:..:argN
 */

char *create_host_list_mpmd(process_groups * pglist, process * plist)
{
    int k, n;
    char *host_list = NULL;
    for (k = 0; k < pglist->npgs; k++) {
        /* Make a list of hosts, the number of processes on each host and the executable and args. */
        /* NOTE: RFCs do not allow : or ; in hostnames */
        if (host_list)
            host_list = mkstr("%s:%s:%d", host_list, pglist->data[k].hostname, pglist->data[k].npids);
        else
            host_list = mkstr("%s:%d", pglist->data[k].hostname, pglist->data[k].npids);
        if (!host_list) {
            error("ALLOCATION ERROR IN BUILD HOST_LIST \n");
        }
        for (n = 0; n < pglist->data[k].npids; n++) {
            host_list = mkstr("%s:%d", host_list, pglist->data[k].plist_indices[n]);

            if (!host_list) {

                error("ALLOCATION ERROR IN BUILD HOST_LIST \n");
            }
        }
        //We use the first of the plist in the group. In each group of processes the exe is the same.
        int plist_index = pglist->data[k].plist_indices[0];
        //We put in host_list the executable:the number of argument
        int argc = plist[plist_index].argc;

        host_list = mkstr("%s:%s:%d", host_list, plist[plist_index].executable_name, argc);
        //Now we put in the host_list for each exe the arguments

        if (plist[plist_index].executable_args != NULL) {

            char **tokenized = tokenize(plist[plist_index].executable_args, " ");
            for (n = 0; n < argc - 1; n++)
                host_list = mkstr("%s:%s", host_list, tokenized[n]);
        }

    }
    return host_list;

}

/**
 * Utility function used to extract the tokens of a string.
 */
char **tokenize(char *line, char *delim)
{
    int argc = 0;
    char *tmp = strdup(line);
    char **argv = (char **) calloc(++argc, sizeof(char *));
    argv[argc - 1] = strtok(tmp, delim);
    argv = (char **) realloc(argv, ++argc * sizeof(char *));
    while ((argv[argc - 1] = strtok(NULL, delim)) != NULL)
        argv = (char **) realloc(argv, ++argc * sizeof(char *));

    return argv;
}

/**
 * This function parses the host list which is an argument of mpispawn.
 * The host_list has the following form: host1:numProc:pid1:pid2..:pidN:exe:argc:arg1:..:argN
 */

/*void parse_host_list_mpmd(int i, int mt_nnodes, char *host_list, char **host, int** ranks, int *np, char **exe, char ***argv)
{
    int j = 0, k;
     while (i > 0) {
                if (i == mt_nnodes)
                    host[j] = strtok (host_list, ":");
                else
                    host[j] = strtok (NULL, ":");
                np[j] = atoi (strtok (NULL, ":"));

                ranks[j] = (int *) malloc (np[j] * sizeof (int));
                for (k = 0; k < np[j]; k++) {
                    ranks[j][k] = atoi (strtok (NULL, ":"));
                }
                //After obtained the ranks we have to obtain the executable
                exe[j] = strtok (NULL, ":");
                int argc = atoi (strtok (NULL, ":"));
                if ( argc >1 )
                    argv[j] = strtok (NULL, ":");
                i--;
                j++;
            }
}*/
