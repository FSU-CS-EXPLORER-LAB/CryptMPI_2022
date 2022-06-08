## -*- Mode: Makefile; -*-
## vim: set ft=automake :
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


if BUILD_NEMESIS_NETMOD_IB

AM_CPPFLAGS += -DNEMESIS_BUILD \
	       -I$(top_srcdir)/src/mpid/ch3/channels/nemesis/netmod/ib \
	       -I$(top_builddir)/src/mpid/ch3/channels/nemesis/netmod/ib

mpi_core_sources +=				\
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_init.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_ckpt.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_connect_to_root.c    \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_finalize.c       \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_poll.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_vc.c             \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_send.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_param.c          \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_cm.c             \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_cell.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_hca.c            \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_process.c        \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_vbuf.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_channel_manager.c    \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_recv.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_srq.c            \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_ds_hash.c        \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_ds_queue.c       \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_errors.c         \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_rdma.c           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_lmt_recv.c       \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_lmt_send.c       \
    src/mpid/ch3/channels/common/src/reg_cache/dreg.c           \
    src/mpid/ch3/channels/common/src/reg_cache/avl.c            \
    src/mpid/ch3/channels/common/src/memory/mem_hooks.c			\
    src/mpid/ch3/channels/common/src/memory/ptmalloc2/mvapich_malloc.c \
    src/mpid/ch3/channels/common/src/util/mv2_utils.c			\
    src/mpid/ch3/channels/common/src/qos/rdma_3dtorus.c			\
    src/mpid/ch3/channels/common/src/detect/arch/mv2_arch_detect.c 	\
    src/mpid/ch3/channels/common/src/detect/hca/mv2_hca_detect.c

noinst_HEADERS +=                                               \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_cell.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_channel_manager.h    \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_cm.h             \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_device.h         \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_ds_hash.h        \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_ds_queue.h       \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_errors.h         \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_finalize.h       \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_hca.h            \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_init.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_lmt.h            \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_poll.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_process.h        \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_rdma.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_recv.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_send.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_srq.h            \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_vbuf.h           \
    src/mpid/ch3/channels/nemesis/netmod/ib/ib_vc.h 

endif BUILD_NEMESIS_NETMOD_IB

