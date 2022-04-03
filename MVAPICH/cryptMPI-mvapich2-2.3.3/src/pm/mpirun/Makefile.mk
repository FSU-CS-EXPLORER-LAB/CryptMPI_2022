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

if BUILD_PM_MPIRUN

include $(top_srcdir)/src/pm/mpirun/src/Makefile.mk

bin_PROGRAMS += src/pm/mpirun/mpirun_rsh \
			    src/pm/mpirun/mpiexec.mpirun_rsh  \
                src/pm/mpirun/mpispawn

AM_CPPFLAGS += -I$(top_srcdir)/src/mpid/ch3/channels/common/include

if WANT_RDYNAMIC
src_pm_mpirun_mpirun_rsh_LDFLAGS = -rdynamic
src_pm_mpirun_mpiexec_mpirun_rsh_LDFLAGS = -rdynamic
src_pm_mpirun_mpispawn_LDFLAGS = -rdynamic
endif

if WANT_CKPT_RUNTIME
bin_PROGRAMS += src/pm/mpirun/mv2_trigger
bin_SCRIPTS += src/pm/mpirun/mv2_checkpoint
endif

src_pm_mpirun_mpirun_rsh_SOURCES =  \
	src/pm/mpirun/mpirun_rsh.c    \
	src/pm/mpirun/mpirun_util.c   \
	src/pm/mpirun/mpmd.c          \
	src/pm/mpirun/mpirun_dbg.c    \
	src/pm/mpirun/mpirun_ckpt.c   \
	src/pm/mpirun/mpirun_params.c \
	src/pm/mpirun/param.c         \
	src/pm/mpirun/signal_processor.c  \
	src/pm/mpirun/wfe_mpirun.c    \
	src/pm/mpirun/m_state.c       \
	src/pm/mpirun/read_specfile.c \
	src/pm/mpirun/gethostip.c     \
	src/pm/mpirun/environ.c

src_pm_mpirun_mpirun_rsh_LDADD = -lm       \
	src/mpid/ch3/channels/common/src/util/mv2_config.o    \
	src/mpid/ch3/channels/common/src/util/crc32h.o        \
	src/mpid/ch3/channels/common/src/util/error_handling.o \
	src/mpid/ch3/channels/common/src/util/debug_utils.o   \
	src/pm/mpirun/src/hostfile/libhostfile.a  \
	src/pm/mpirun/src/db/libdb.a

src_pm_mpirun_mpiexec_mpirun_rsh_SOURCES =     \
	src/pm/mpirun/mpirun_rsh.c    \
	src/pm/mpirun/mpirun_util.c   \
	src/pm/mpirun/mpmd.c          \
	src/pm/mpirun/mpirun_dbg.c    \
	src/pm/mpirun/mpirun_ckpt.c   \
	src/pm/mpirun/mpirun_params_comp.c    \
	src/pm/mpirun/param.c         \
	src/pm/mpirun/signal_processor.c  \
	src/pm/mpirun/wfe_mpirun.c    \
	src/pm/mpirun/m_state.c       \
	src/pm/mpirun/read_specfile.c \
	src/pm/mpirun/gethostip.c     \
	src/pm/mpirun/environ.c

src_pm_mpirun_mpiexec_mpirun_rsh_LDADD = -lm 	\
	src/mpid/ch3/channels/common/src/util/mv2_config.o    \
	src/mpid/ch3/channels/common/src/util/crc32h.o        \
	src/mpid/ch3/channels/common/src/util/error_handling.o \
	src/mpid/ch3/channels/common/src/util/debug_utils.o   \
	src/pm/mpirun/src/hostfile/libhostfile.a  	\
	src/pm/mpirun/src/db/libdb.a              	\
	src/pm/mpirun/src/slurm/libslurm.a 		  	\
	src/pm/mpirun/src/slurm/libnodelist.a 		\
	src/pm/mpirun/src/slurm/libtasklist.a 		\
	src/pm/mpirun/src/pbs/libpbs.a

src_pm_mpirun_mpispawn_SOURCES =    \
	src/pm/mpirun/mpispawn.c      \
	src/pm/mpirun/mpirun_util.c   \
	src/pm/mpirun/mpispawn_tree.c \
	src/pm/mpirun/pmi_tree.c      \
	src/pm/mpirun/mpmd.c          \
	src/pm/mpirun/opt.c           \
	src/pm/mpirun/crfs.c          \
	src/pm/mpirun/log.c           \
	src/pm/mpirun/crfs_ib.c       \
	src/pm/mpirun/ib_buf.c        \
	src/pm/mpirun/ib_comm.c       \
	src/pm/mpirun/thread_pool.c   \
	src/pm/mpirun/ibutil.c        \
	src/pm/mpirun/bitmap.c        \
	src/pm/mpirun/work_queue.c    \
	src/pm/mpirun/openhash.c      \
	src/pm/mpirun/ckpt_file.c     \
	src/pm/mpirun/genhash.c       \
	src/pm/mpirun/crfs_wa.c       \
	src/pm/mpirun/mpispawn_ckpt.c \
	src/pm/mpirun/signal_processor.c  \
	src/pm/mpirun/gethostip.c     \
	src/pm/mpirun/environ.c

src_pm_mpirun_mpispawn_LDADD = -lm -lpthread 	\
	src/mpid/ch3/channels/common/src/util/error_handling.o 	\
	src/mpid/ch3/channels/common/src/util/debug_utils.o 	\
	src/pm/mpirun/src/db/libdb.a

src_pm_mpirun_mv2_trigger_SOURCES = src/pm/mpirun/mv2_trigger.c
src_pm_mpirun_mv2_trigger_LDADD = -lpthread

endif BUILD_PM_MPIRUN
