AM_CPPFLAGS = -I$(top_srcdir)/../../mpid/ch3/channels/common/include
AM_CPPFLAGS += -I$(top_srcdir) -I$(srcdir)/../

libnodelist_a_YFLAGS = -d -p nodelist_yy
libtasklist_a_YFLAGS = -d -p tasklist_yy

noinst_LIBRARIES = libslurm.a libnodelist.a libtasklist.a

BUILT_SOURCES = libnodelist_a-nodelist_parser.h libtasklist_a-tasklist_parser.h

libslurm_a_SOURCES = slurm_startup.c
libnodelist_a_SOURCES = nodelist_parser.y nodelist_scanner.l suffixlist.c
libtasklist_a_SOURCES = tasklist_parser.y tasklist_scanner.l
