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

#include <parser.h>
#include <string.h>
%}

%option noyywrap
%option nounput
%option noinput

decimal     ([[:digit:]]+)
whitespace  ([ \t]+)
text        ([^[:space:]:#]+)
comment     (#.*)

%option prefix="hostfile_yy" outfile="lex.yy.c"

%%

{comment}|{whitespace}  /* eat up comments and whitespace */

{decimal}               { hostfile_yylval.decimal = (size_t)atoi(yytext);
                          return DECIMAL; }
{text}                  { hostfile_yylval.text = strdup(yytext); return TEXT; }
:                       { return ':'; }
\n                      { return '\n'; }

