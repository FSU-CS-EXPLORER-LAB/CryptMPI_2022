## Copyright (c) 2001-2019, The Ohio State University. All rights
## reserved.
##
## This file is part of the MVAPICH2 software package developed by the
## team members of The Ohio State University's Network-Based Computing
## Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
##
## For detailed copyright and licensing information, please refer to the
## copyright file COPYRIGHT in the top level MVAPICH2 directory.
##

## todo: provide configure args for path to nvcc and specification of -arch or
## 	-maxrregcount values

NVCC = nvcc
NVCFLAGS = -cuda -ccbin $(CXX)

SUFFIXES += .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@ $<

noinst_LTLIBRARIES += lib/lib@MPILIBNAME@_cuda_osu.la
lib_lib@MPILIBNAME@_cuda_osu_la_SOURCES =                   \
	    src/mpid/ch3/channels/mrail/src/cuda/pack_unpack.cu

lib_lib@MPILIBNAME@_cuda_osu_la_CXXFLAGS = $(AM_CXXFLAGS)
### use extra flags if host compiler is PGI
if BUILD_USE_PGI
lib_lib@MPILIBNAME@_cuda_osu_la_CXXFLAGS += --nvcchost --no_preincludes
endif
lib_lib@MPILIBNAME@_cuda_osu_la_LIBADD = -lstdc++

lib_lib@MPILIBNAME@_la_LIBADD += lib/lib@MPILIBNAME@_cuda_osu.la

CLEANFILES += src/mpid/ch3/channels/mrail/src/cuda/*.cpp
