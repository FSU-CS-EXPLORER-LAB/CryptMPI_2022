%{
/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#include <debug_utils.h>
#include <db/text.h>
#include <suffixlist.h>

#include <stdio.h>
#include <stdlib.h>

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yylex();
void yyerror (char const *);
extern YY_BUFFER_STATE nodelist_yy_scan_string (char const *);
extern void nodelist_yy_delete_buffer (YY_BUFFER_STATE);

static void slurm_add_hostname(const char *);
static void slurm_add_hostnames(const char *, sl_handle);
%}

%union {
    char * text;
    sl_handle sl;
    sr_handle sr;
}

%{
#if YYDEBUG
static void print_token_value (FILE *, int, YYSTYPE);
#define YYPRINT(file, type, value) print_token_value (file, type, value)
#endif
%}

    
%token <text> TEXT
%type <sl> suffixlist
%type <sr> suffixrange
%type <text> hostname prefix suffix

%%

hostlist:   hostrange
        |   hostlist ',' hostrange
;

hostrange:  hostname                    { slurm_add_hostname($1); }
         |  prefix '[' suffixlist ']'   { slurm_add_hostnames($1, $3); }
;

suffixlist: suffixrange                 { $$ = sl_create($1); }
          | suffixlist ',' suffixrange  { $$ = sl_insert($1, $3); }
;

suffixrange:    suffix                  { $$ = sr_create($1, $1); }
           |    suffix '-' suffix       { $$ = sr_create($1, $3); }
;

hostname:   TEXT
;

prefix: TEXT
;

suffix: TEXT
;

%%

static char (* nodelist)[256];
static size_t num_nodes;
static size_t nodelist_index;

static void
slurm_add_hostname(const char * hostname)
{
    strcpy(nodelist[nodelist_index++], hostname); 
}

static void
slurm_add_hostnames(const char * prefix, sl_handle sl)
{
    char hostname[255], * suffix;
    sl_iterator sli = sli_create(sl);

    strcpy(hostname, prefix);
    suffix = hostname + strlen(hostname);

    while (sli_hasNext(sli)) {
        *suffix = '\0';
        strcat(hostname, sli_next(sli));
        slurm_add_hostname(hostname);
    }
}

/*
 * Given a slurm nodelist (str) with number of nodes (nnodes) populate
 * list[] with the hostname of each node.  Returns the number of nodes
 * successfully processed.
 */
extern int
slurm_init_nodelist (char const * const nodelist_spec, size_t nnodes, char
    nodes[nnodes][256])
{
    YY_BUFFER_STATE buffer_state = nodelist_yy_scan_string(nodelist_spec);
    int exit_code;

    nodelist = nodes;
    num_nodes = nnodes;
    nodelist_index = 0;

    exit_code = yyparse();
    nodelist_yy_delete_buffer(buffer_state);

    return exit_code;
}

void
yyerror (char const * s)
{
    PRINT_ERROR("Error parsing slurm nodelist `%s'\n", s);
}

#if YYDEBUG
static void
print_token_value (FILE * file, int type, YYSTYPE value)
{
    switch (type) {
        case TEXT:
            fprintf(file, "%s", value.text);
            break;
    }
}
#endif
