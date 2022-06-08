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

AM_CPPFLAGS += -D_GNU_SOURCE

mpi_core_sources	+=					\
    src/mpid/ch3/channels/mrail/src/rdma/mpid_mrail_rndv.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_finalize.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_init.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_isend.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_isendv.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_istartmsg.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_istartmsgv.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_request.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_progress.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_cancel_send.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_read_progress.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_comm_spawn_multiple.c	\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_comm_accept.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_comm_connect.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_open_port.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_abort.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_istartrndvmsg.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_packetizedtransfer.c	\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_rndvtransfer.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_smp_progress.c		\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_get_business_card.c	\
    src/mpid/ch3/channels/mrail/src/rdma/ch3i_comm.c			\
    src/mpid/ch3/channels/mrail/src/rdma/ch3_contigsend.c       \
    src/mpid/ch3/channels/mrail/src/rdma/ch3_win_fns.c			\
	src/mpid/ch3/channels/mrail/src/rdma/ibv_sharp.c

mpi_convenience_libs += libch3affinity.la

if BUILD_MRAIL_GEN2

AM_CPPFLAGS += -I$(top_srcdir)/src/mpid/ch3/channels/mrail/src/gen2 \
			   -I$(top_srcdir)/src/mpi/coll
AM_CPPFLAGS += -I$(top_srcdir)/src/mpi/romio/adio/include

mpi_core_sources	+=					\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_send.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_recv.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_ud.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_ud_zcopy.c			\
    src/mpid/ch3/channels/mrail/src/gen2/rdma_iba_init.c		\
    src/mpid/ch3/channels/mrail/src/gen2/rdma_iba_priv.c		\
    src/mpid/ch3/channels/common/src/reg_cache/dreg.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_param.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_env_params.c		\
    src/mpid/ch3/channels/mrail/src/gen2/vbuf.c				\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_channel_manager.c		\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_rma.c			\
    src/mpid/ch3/channels/mrail/src/gen2/rdma_iba_1sc.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_rndv.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_priv.c			\
    src/mpid/ch3/channels/common/src/reg_cache/avl.c		\
    src/mpid/ch3/channels/common/src/cm/cm.c				\
    src/mpid/ch3/channels/common/src/rdma_cm/rdma_cm.c		\
    src/mpid/ch3/channels/mrail/src/gen2/ring_startup.c				\
    src/mpid/ch3/channels/mrail/src/gen2/sysreport.c				\
    src/mpid/ch3/channels/common/src/detect/arch/mv2_arch_detect.c 	\
    src/mpid/ch3/channels/common/src/detect/hca/mv2_hca_detect.c 	\
    src/mpid/ch3/channels/common/src/memory/mem_hooks.c			\
    src/mpid/ch3/channels/common/src/memory/ptmalloc2/mvapich_malloc.c \
    src/mpid/ch3/channels/common/src/util/mv2_utils.c			\
    src/mpid/ch3/channels/common/src/ud-hybrid/mv2_ud_init.c	\
    src/mpid/ch3/channels/common/src/qos/rdma_3dtorus.c			\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_cuda_rndv.c		\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_cuda_util.c		\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_cuda_event.c		\
    src/mpid/ch3/channels/mrail/src/gen2/ibv_cuda_ipc.c			\
    src/mpid/ch3/channels/common/src/mcast/ibv_mcast.c			\
	src/mpid/ch3/channels/mrail/src/gen2/mv2_mpit_cvars.c

endif

if BUILD_MRAIL_CUDA_KERNELS
include $(top_srcdir)/src/mpid/ch3/channels/mrail/src/cuda/Makefile.mk
endif
