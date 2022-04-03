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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yylex();
void yyerror (char const *);
extern YY_BUFFER_STATE tasklist_yy_scan_string (char const *);
extern void tasklist_yy_delete_buffer (YY_BUFFER_STATE);

static void slurm_add_ntask(size_t);
static void slurm_add_ntasks(size_t, size_t);

#define YYSTYPE size_t 

#if YYDEBUG
static void print_token_value (FILE *, int, YYSTYPE);
#define YYPRINT(file, type, value) print_token_value (file, type, value)
#endif
%}

%token DECIMAL

%%

tasklist:   taskrange
        |   tasklist ',' taskrange
;

taskrange:  ntasks                          { slurm_add_ntask($1); }
         |  ntasks '(' 'x' multiplier ')'   { slurm_add_ntasks($1, $4); }
;

ntasks:     DECIMAL
;

multiplier: DECIMAL
;

%%

static size_t (* tasklist)[];
static size_t num_nodes;
static size_t tasklist_index;

static void
slurm_add_ntask(size_t ntasks)
{
    (*tasklist)[tasklist_index++] = ntasks;
}

static void
slurm_add_ntasks(size_t ntasks, size_t multiplier)
{
    while (multiplier--) {
        slurm_add_ntask(ntasks);
    }
}

/*
 * Given a slurm tasklist (str) with number of nodes (nnodes) populate
 * list[] with the ntasks of each node.  Returns the number of nodes
 * successfully processed.
 */
extern int 
slurm_init_tasklist (char const * const tasklist_spec, size_t nnodes, size_t
    (* ntasks)[nnodes])
{
    YY_BUFFER_STATE buffer_state = tasklist_yy_scan_string(tasklist_spec);
    int exit_code;

    tasklist = ntasks;
    num_nodes = nnodes;
    tasklist_index = 0;

    exit_code = yyparse();
    tasklist_yy_delete_buffer(buffer_state);

    return exit_code;
}

void
yyerror (char const * s)
{
    PRINT_ERROR("Error parsing slurm tasks per node `%s'\n", s);
}

#if YYDEBUG
static void
print_token_value (FILE * file, int type, YYSTYPE value)
{
    switch (type) {
        case DECIMAL:
            fprintf(file, "%d", (int)value);
            break;
    }
}
#endif
