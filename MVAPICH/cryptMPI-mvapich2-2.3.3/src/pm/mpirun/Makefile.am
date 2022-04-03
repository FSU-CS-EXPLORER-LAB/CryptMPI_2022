SUBDIRS = src
bin_PROGRAMS = mpirun_rsh mpiexec.mpirun_rsh mpispawn

AM_CPPFLAGS = -I$(top_srcdir)/../../mpid/ch3/channels/common/include

if WANT_RDYNAMIC
mpirun_rsh_LDFLAGS = -rdynamic
mpiexec_mpirun_rsh_LDFLAGS = -rdynamic
mpispawn_LDFLAGS = -rdynamic
endif

if WANT_CKPT_RUNTIME
bin_PROGRAMS += mv2_trigger
dist_bin_SCRIPTS = mv2_checkpoint
endif

mpirun_rsh_SOURCES = mpirun_rsh.c mpirun_util.c mpmd.c mpirun_dbg.c mpirun_ckpt.c mpirun_params.c param.c signal_processor.c wfe_mpirun.c m_state.c read_specfile.c gethostip.c
mpirun_rsh_LDADD = -lm ../../mpid/ch3/channels/common/src/util/mv2_config.o ../../mpid/ch3/channels/common/src/util/crc32h.o ../../mpid/ch3/channels/common/src/util/error_handling.o ../../mpid/ch3/channels/common/src/util/debug_utils.o src/hostfile/libhostfile.a src/db/libdb.a

mpiexec_mpirun_rsh_SOURCES = mpirun_rsh.c mpirun_util.c mpmd.c mpirun_dbg.c mpirun_ckpt.c mpirun_params_comp.c param.c signal_processor.c wfe_mpirun.c m_state.c read_specfile.c gethostip.c
mpiexec_mpirun_rsh_LDADD = -lm ../../mpid/ch3/channels/common/src/util/mv2_config.o ../../mpid/ch3/channels/common/src/util/crc32h.o ../../mpid/ch3/channels/common/src/util/error_handling.o ../../mpid/ch3/channels/common/src/util/debug_utils.o src/hostfile/libhostfile.a src/db/libdb.a src/slurm/libslurm.a src/slurm/libnodelist.a src/slurm/libtasklist.a src/pbs/libpbs.a

mpispawn_SOURCES = mpispawn.c mpirun_util.c mpispawn_tree.c pmi_tree.c mpmd.c opt.c crfs.c log.c crfs_ib.c ib_buf.c ib_comm.c thread_pool.c ibutil.c bitmap.c  work_queue.c openhash.c ckpt_file.c genhash.c crfs_wa.c mpispawn_ckpt.c signal_processor.c gethostip.c
mpispawn_LDADD = -lm -lpthread ../../mpid/ch3/channels/common/src/util/error_handling.o ../../mpid/ch3/channels/common/src/util/debug_utils.o src/db/libdb.a

mv2_trigger_SOURCES = mv2_trigger.c 
mv2_trigger_LDADD = -lpthread

install-alt: install
mpich2-build-install: install
mpich2-build-uninstall: uninstall

mandoc:

htmldoc:

latexdoc:

