## -*- Mode: Makefile; -*-
## vim: set ft=automake :
##
## (C) 2011 by Argonne National Laboratory.
##     See COPYRIGHT in top-level directory.
##
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

# mpi_sources includes only the routines that are MPI function entry points
# The code for the MPI operations (e.g., MPI_SUM) is not included in 
# mpi_sources
mpi_sources +=                     \
    src/mpi/coll/allreduce.c       \
    src/mpi/coll/barrier.c         \
    src/mpi/coll/op_create.c       \
    src/mpi/coll/op_free.c         \
    src/mpi/coll/bcast.c           \
    src/mpi/coll/alltoall.c        \
    src/mpi/coll/alltoallv.c       \
    src/mpi/coll/reduce.c          \
    src/mpi/coll/scatter.c         \
    src/mpi/coll/gather.c          \
    src/mpi/coll/scatterv.c        \
    src/mpi/coll/gatherv.c         \
    src/mpi/coll/scan.c            \
    src/mpi/coll/exscan.c          \
    src/mpi/coll/allgather.c       \
    src/mpi/coll/allgatherv.c      \
    src/mpi/coll/red_scat.c        \
    src/mpi/coll/alltoallw.c       \
    src/mpi/coll/reduce_local.c    \
    src/mpi/coll/op_commutative.c  \
    src/mpi/coll/red_scat_block.c  \
    src/mpi/coll/iallgather.c      \
    src/mpi/coll/iallgatherv.c     \
    src/mpi/coll/iallreduce.c      \
    src/mpi/coll/ialltoall.c       \
    src/mpi/coll/ialltoallv.c      \
    src/mpi/coll/ialltoallw.c      \
    src/mpi/coll/ibarrier.c        \
    src/mpi/coll/ibcast.c          \
    src/mpi/coll/iexscan.c         \
    src/mpi/coll/igather.c         \
    src/mpi/coll/igatherv.c        \
    src/mpi/coll/ired_scat.c       \
    src/mpi/coll/ired_scat_block.c \
    src/mpi/coll/ireduce.c         \
    src/mpi/coll/iscan.c           \
    src/mpi/coll/iscatter.c        \
    src/mpi/coll/iscatterv.c

if BUILD_OSU_MVAPICH
mpi_sources +=                     \
    src/mpi/coll/iallgather_osu.c  \
    src/mpi/coll/iallreduce_osu.c  \
    src/mpi/coll/ibarrier_osu.c    \
    src/mpi/coll/iallgatherv_osu.c \
    src/mpi/coll/ialltoall_osu.c   \
    src/mpi/coll/ialltoallv_osu.c  \
    src/mpi/coll/ibcast_osu.c      \
    src/mpi/coll/igather_osu.c     \
    src/mpi/coll/ired_scat_osu.c   \
    src/mpi/coll/ireduce_osu.c     \
    src/mpi/coll/iscatter_osu.c    \
    src/mpi/coll/allgather_osu.c   \
    src/mpi/coll/allgatherv_osu.c  \
    src/mpi/coll/allreduce_osu.c   \
    src/mpi/coll/alltoall_osu.c    \
    src/mpi/coll/alltoallv_osu.c   \
    src/mpi/coll/barrier_osu.c     \
    src/mpi/coll/bcast_osu.c       \
    src/mpi/coll/gather_osu.c      \
    src/mpi/coll/gatherv_osu.c     \
    src/mpi/coll/reduce_osu.c      \
    src/mpi/coll/scatter_osu.c     \
    src/mpi/coll/scan_osu.c        \
    src/mpi/coll/exscan_osu.c      \
    src/mpi/coll/red_scat_osu.c    \
    src/mpi/coll/red_scat_block_osu.c    \
    src/mpi/coll/ch3_shmem_coll.c  \
    src/mpi/coll/alltoall_cuda_osu.c \
    src/mpi/coll/allgather_cuda_osu.c\
    src/mpi/coll/reduce_tuning.c   \
    src/mpi/coll/allgather_tuning.c\
    src/mpi/coll/iallgather_tuning.c\
    src/mpi/coll/iallreduce_tuning.c\
    src/mpi/coll/ibarrier_tuning.c \
    src/mpi/coll/iallgatherv_tuning.c\
    src/mpi/coll/red_scat_tuning.c \
    src/mpi/coll/red_scat_block_tuning.c \
    src/mpi/coll/allgatherv_tuning.c\
    src/mpi/coll/alltoall_tuning.c  \
    src/mpi/coll/alltoallv_tuning.c  \
    src/mpi/coll/allreduce_tuning.c\
    src/mpi/coll/ireduce_tuning.c  \
    src/mpi/coll/ired_scat_tuning.c\
    src/mpi/coll/ialltoall_tuning.c\
    src/mpi/coll/ialltoallv_tuning.c\
    src/mpi/coll/bcast_tuning.c    \
    src/mpi/coll/ibcast_tuning.c   \
    src/mpi/coll/gather_tuning.c   \
    src/mpi/coll/igather_tuning.c  \
    src/mpi/coll/scatter_tuning.c  \
    src/mpi/coll/iscatter_tuning.c
endif

mpi_core_sources += \
    src/mpi/coll/allred_group.c   \
    src/mpi/coll/barrier_group.c  \
    src/mpi/coll/helper_fns.c     \
    src/mpi/coll/opsum.c          \
    src/mpi/coll/opmax.c          \
    src/mpi/coll/opmin.c          \
    src/mpi/coll/opband.c         \
    src/mpi/coll/opbor.c          \
    src/mpi/coll/opbxor.c         \
    src/mpi/coll/opland.c         \
    src/mpi/coll/oplor.c          \
    src/mpi/coll/oplxor.c         \
    src/mpi/coll/opprod.c         \
    src/mpi/coll/opminloc.c       \
    src/mpi/coll/opmaxloc.c       \
    src/mpi/coll/opno_op.c        \
    src/mpi/coll/opreplace.c      \
    src/mpi/coll/nbcutil.c

noinst_HEADERS +=           \
    src/mpi/coll/collutil.h

