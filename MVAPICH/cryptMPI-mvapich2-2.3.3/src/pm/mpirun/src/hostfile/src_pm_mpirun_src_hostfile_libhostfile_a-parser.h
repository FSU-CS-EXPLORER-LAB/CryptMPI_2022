/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_HOSTFILE_YY_SRC_PM_MPIRUN_SRC_HOSTFILE_SRC_PM_MPIRUN_SRC_HOSTFILE_LIBHOSTFILE_A_PARSER_H_INCLUDED
# define YY_HOSTFILE_YY_SRC_PM_MPIRUN_SRC_HOSTFILE_SRC_PM_MPIRUN_SRC_HOSTFILE_LIBHOSTFILE_A_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int hostfile_yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    DECIMAL = 258,
    TEXT = 259
  };
#endif
/* Tokens.  */
#define DECIMAL 258
#define TEXT 259

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 46 "src/pm/mpirun/src/hostfile/parser.y" /* yacc.c:1909  */

    size_t decimal;
    char * text;

#line 67 "src/pm/mpirun/src/hostfile/src_pm_mpirun_src_hostfile_libhostfile_a-parser.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE hostfile_yylval;

int hostfile_yyparse (void);

#endif /* !YY_HOSTFILE_YY_SRC_PM_MPIRUN_SRC_HOSTFILE_SRC_PM_MPIRUN_SRC_HOSTFILE_LIBHOSTFILE_A_PARSER_H_INCLUDED  */
