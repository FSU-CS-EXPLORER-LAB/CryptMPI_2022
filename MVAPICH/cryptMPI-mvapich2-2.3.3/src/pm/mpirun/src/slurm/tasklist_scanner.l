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

#include <libtasklist_a-tasklist_parser.h>
#include <string.h>
%}

%option noyywrap
%option nounput
%option noinput

decimal      ([[:digit:]]+)

%option prefix="tasklist_yy" outfile="lex.yy.c"

%x range

%%

,               { return ','; }
\(              { return '('; }
\)              { return ')'; }
x               { return 'x'; }
{decimal}       { tasklist_yylval = atoi(yytext); return DECIMAL; }
