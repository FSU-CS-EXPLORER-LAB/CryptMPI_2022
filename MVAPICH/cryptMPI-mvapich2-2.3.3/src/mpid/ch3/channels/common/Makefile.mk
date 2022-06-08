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


if BUILD_LIB_CH3AFFINITY
noinst_LTLIBRARIES += libch3affinity.la
libch3affinity_la_SOURCES = src/mpid/ch3/channels/common/src/affinity/hwloc_bind.c
endif

if BUILD_LIB_SCR
EXTRA_DIST += src/mpid/ch3/channels/common/src/scr/LICENSE.TXT
AM_CPPFLAGS += -I$(top_srcdir)/src/mpid/ch3/channels/common/src/scr
noinst_LTLIBRARIES += libscr_base.la libscr.la

include_HEADERS += \
	src/mpid/ch3/channels/common/src/scr/scr.h \
	src/mpid/ch3/channels/common/src/scr/scrf.h

noinst_HEADERS += \
	src/mpid/ch3/channels/common/src/scr/scr_conf.h \
	src/mpid/ch3/channels/common/src/scr/scr_globals.h \
	src/mpid/ch3/channels/common/src/scr/scr_err.h \
	src/mpid/ch3/channels/common/src/scr/scr_io.h \
	src/mpid/ch3/channels/common/src/scr/scr_path.h \
	src/mpid/ch3/channels/common/src/scr/scr_path_mpi.h \
	src/mpid/ch3/channels/common/src/scr/scr_compress.h \
	src/mpid/ch3/channels/common/src/scr/scr_util.h \
	src/mpid/ch3/channels/common/src/scr/scr_util_mpi.h \
	src/mpid/ch3/channels/common/src/scr/scr_split.h \
	src/mpid/ch3/channels/common/src/scr/scr_hash.h \
	src/mpid/ch3/channels/common/src/scr/scr_hash_util.h \
	src/mpid/ch3/channels/common/src/scr/scr_hash_mpi.h \
	src/mpid/ch3/channels/common/src/scr/scr_flush_file_mpi.h \
	src/mpid/ch3/channels/common/src/scr/tv_data_display.h \
	src/mpid/ch3/channels/common/src/scr/scr_filemap.h \
	src/mpid/ch3/channels/common/src/scr/scr_index_api.h \
	src/mpid/ch3/channels/common/src/scr/scr_meta.h \
	src/mpid/ch3/channels/common/src/scr/scr_dataset.h \
	src/mpid/ch3/channels/common/src/scr/scr_storedesc.h \
	src/mpid/ch3/channels/common/src/scr/scr_groupdesc.h \
	src/mpid/ch3/channels/common/src/scr/scr_reddesc.h \
	src/mpid/ch3/channels/common/src/scr/scr_reddesc_apply.h \
	src/mpid/ch3/channels/common/src/scr/scr_reddesc_recover.h \
	src/mpid/ch3/channels/common/src/scr/scr_summary.h \
	src/mpid/ch3/channels/common/src/scr/scr_cache.h \
	src/mpid/ch3/channels/common/src/scr/scr_cache_rebuild.h \
	src/mpid/ch3/channels/common/src/scr/scr_fetch.h \
	src/mpid/ch3/channels/common/src/scr/scr_flush.h \
	src/mpid/ch3/channels/common/src/scr/scr_flush_sync.h \
	src/mpid/ch3/channels/common/src/scr/scr_flush_async.h \
	src/mpid/ch3/channels/common/src/scr/scr_config.h \
	src/mpid/ch3/channels/common/src/scr/scr_param.h \
	src/mpid/ch3/channels/common/src/scr/scr_env.h \
	src/mpid/ch3/channels/common/src/scr/scr_log.h \
	src/mpid/ch3/channels/common/src/scr/scr_halt.h \
	src/mpid/ch3/channels/common/src/scr/queue.h

libscr_base_la_SOURCES = \
	src/mpid/ch3/channels/common/src/scr/scr_io.c \
        src/mpid/ch3/channels/common/src/scr/scr_path.c \
        src/mpid/ch3/channels/common/src/scr/scr_compress.c \
        src/mpid/ch3/channels/common/src/scr/scr_util.c \
        src/mpid/ch3/channels/common/src/scr/scr_hash.c \
        src/mpid/ch3/channels/common/src/scr/scr_hash_util.c \
        src/mpid/ch3/channels/common/src/scr/tv_data_display.c \
        src/mpid/ch3/channels/common/src/scr/scr_filemap.c \
        src/mpid/ch3/channels/common/src/scr/scr_index_api.c \
        src/mpid/ch3/channels/common/src/scr/scr_meta.c \
        src/mpid/ch3/channels/common/src/scr/scr_dataset.c \
        src/mpid/ch3/channels/common/src/scr/scr_config.c \
        src/mpid/ch3/channels/common/src/scr/scr_param.c \
        src/mpid/ch3/channels/common/src/scr/scr_env.c \
        src/mpid/ch3/channels/common/src/scr/scr_log.c \
        src/mpid/ch3/channels/common/src/scr/scr_halt.c

libscr_la_SOURCES = \
	src/mpid/ch3/channels/common/src/scr/scr_globals.c \
        src/mpid/ch3/channels/common/src/scr/scr_err_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_path_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_config_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_util_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_split.c \
        src/mpid/ch3/channels/common/src/scr/scr_hash_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_flush_file_mpi.c \
        src/mpid/ch3/channels/common/src/scr/scr_storedesc.c \
        src/mpid/ch3/channels/common/src/scr/scr_groupdesc.c \
        src/mpid/ch3/channels/common/src/scr/scr_reddesc.c \
        src/mpid/ch3/channels/common/src/scr/scr_reddesc_apply.c \
        src/mpid/ch3/channels/common/src/scr/scr_reddesc_recover.c \
        src/mpid/ch3/channels/common/src/scr/scr_summary.c \
        src/mpid/ch3/channels/common/src/scr/scr_cache.c \
        src/mpid/ch3/channels/common/src/scr/scr_cache_rebuild.c \
        src/mpid/ch3/channels/common/src/scr/scr_fetch.c \
        src/mpid/ch3/channels/common/src/scr/scr_flush.c \
        src/mpid/ch3/channels/common/src/scr/scr_flush_sync.c \
        src/mpid/ch3/channels/common/src/scr/scr_flush_async.c \
        src/mpid/ch3/channels/common/src/scr/scr.c \
        src/mpid/ch3/channels/common/src/scr/scrf.c

mpi_convenience_libs += libscr_base.la libscr.la

edit = sed \
       -e 's|@bindir[@]|$(bindir)|g' \
       -e 's|@libdir[@]|$(libdir)|g' \
       -e 's|@pkgdatadir[@]|$(pkgdatadir)|g' \
       -e 's|@prefix[@]|$(prefix)|g' \
       -e 's|@PDSH_EXE[@]|@PDSH_EXE@|g' \
       -e 's|@DSHBAK_EXE[@]|@DSHBAK_EXE@|g'

scr_scripts = $(srcdir)/src/mpid/ch3/channels/common/src/scr/scripts

scr_cntl_dir scr_check_node scr_glob_hosts scr_postrun scr_prerun scr_test_datemanip scr_test_runtime scr_watchdog scr_inspect scr_list_down_nodes scr_scavenge scr_run scr_env scr_halt scr_kill_jobstep scr_get_jobstep_id scr_param.pm: scr.scripts
	rm -f $@ $@.tmp
	srcdir=''; \
	       test -f ./$@.in || srcdir=$(srcdir)/; \
	       $(edit) $${srcdir}$@.in >$@.tmp
	chmod +x $@.tmp
	chmod a-w $@.tmp
	mv $@.tmp $@

scr.scripts: Makefile
	touch scr.scripts
	cp $(scr_scripts)/common/* .
	cp $(scr_scripts)/TLCC/* .

scr_hostlist.pm: scr.scripts
scr_cntl_dir: $(scr_scripts)/common/scr_cntl_dir.in
src_check_node: $(scr_scripts)/common/scr_check_node.in
scr_glob_hosts: $(scr_scripts)/common/scr_glob_hosts.in
scr_postrun: $(scr_scripts)/common/scr_postrun.in
scr_prerun: $(scr_scripts)/common/scr_prerun.in
scr_test_datemanip: $(scr_scripts)/common/scr_test_datemanip.in
scr_test_runtime: $(scr_scripts)/common/scr_test_runtime.in
scr_watchdog: $(scr_scripts)/common/scr_watchdog.in
scr_inspect: $(scr_scripts)/TLCC/scr_inspect.in
scr_list_down_nodes: $(scr_scripts)/TLCC/scr_list_down_nodes.in
scr_scavenge: $(scr_scripts)/TLCC/scr_scavenge.in
scr_run: $(scr_scripts)/TLCC/scr_run.in
scr_env: $(scr_scripts)/TLCC/scr_env.in
scr_halt: $(scr_scripts)/TLCC/scr_halt.in
scr_kill_jobstep: $(scr_scripts)/TLCC/scr_kill_jobstep.in
scr_get_jobstep_id: $(scr_scripts)/TLCC/scr_get_jobstep_id.in
scr_param.pm: $(scr_scripts)/common/scr_param.pm.in

dist_bin_SCRIPTS += \
	scr_cntl_dir \
	scr_check_node \
	scr_glob_hosts \
	scr_postrun \
	scr_prerun \
	scr_test_datemanip \
	scr_test_runtime \
	scr_watchdog \
	scr_inspect \
	scr_list_down_nodes \
	scr_scavenge \
	scr_run \
	scr_env \
	scr_halt \
	scr_kill_jobstep \
	scr_get_jobstep_id 

dist_pkgdata_DATA += \
	scr_param.pm \
	scr_hostlist.pm

CLEANFILES += scr.scripts $(dist_bin_SCRIPTS) $(dist_pkgdata_DATA)
endif

if BUILD_MRAIL

include $(top_srcdir)/src/mpid/ch3/channels/common/src/Makefile.mk

else

if BUILD_CH3_PSM
include $(top_srcdir)/src/mpid/ch3/channels/common/src/Makefile.mk
noinst_LTLIBRARIES += libch3affinity.la
libch3affinity_la_SOURCES = src/mpid/ch3/channels/common/src/affinity/hwloc_bind.c
endif

endif

if BUILD_NEMESIS_NETMOD_IB

include $(top_srcdir)/src/mpid/ch3/channels/common/src/Makefile.mk

endif
