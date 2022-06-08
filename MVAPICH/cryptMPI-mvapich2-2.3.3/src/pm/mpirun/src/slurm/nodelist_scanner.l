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

#include <suffixlist.h>
#include <libnodelist_a-nodelist_parser.h>
#include <string.h>
%}

%option noyywrap
%option nounput
%option noinput

lb          (\[)
rb          (\])
suffix      ([[:digit:]]+)

hex4        ([[:xdigit:]]{1,4})
hexseq      ({hex4}(:{hex4}*))
hexpart     ({hexseq}|({hexseq}::({hexseq}?))|::{hexseq})
IPv4address (([[:digit:]]{1,3}"."){3}([[:digit:]]{1,3}))
IPv6address ({hexpart}(":"{IPv4address})?)

labelA      ([[:alnum:]]([-[:alnum:]]*[[:alnum:]])?)
labelB      ([[:alpha:]]([-[:alnum:]]*[[:alnum:]])?)
hostname    ({labelA}(\.{labelB})*)


%option prefix="nodelist_yy" outfile="lex.yy.c"

%x range

%%

<INITIAL,range>,    { return ','; }
{lb}                { BEGIN(range); return '['; }
<range>{rb}         { BEGIN(INITIAL); return ']'; }
<range>-            { return '-'; }
<range>{suffix}     { nodelist_yylval.text = strdup(yytext); return TEXT; }
{IPv4address}       { nodelist_yylval.text = strdup(yytext); return TEXT; }
{IPv6address}       { nodelist_yylval.text = strdup(yytext); return TEXT; }
{hostname}          { nodelist_yylval.text = strdup(yytext); return TEXT; }

