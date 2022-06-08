EXTRA_DIST = COPYRIGHT README limic.spec module/Makefile module/limic.c\
	     module/limic.h module/limic module/10-limic.rules
SUBDIRS = src

if BUILD_KMOD
udevdir = @sysconfdir@/udev/rules.d
initdir = @sysconfdir@/init.d

init_SCRIPTS = module/limic
udev_DATA = module/10-limic.rules

module/limic.ko:
	test -f $(builddir)/module/Makefile ||\
	    $(LN_S) $(abs_top_srcdir)/module/Makefile $(builddir)/module/
	test -f $(builddir)/module/limic.c ||\
	    $(LN_S) $(abs_top_srcdir)/module/limic.c $(builddir)/module/
	test -f $(builddir)/module/limic.h ||\
	    $(LN_S) $(abs_top_srcdir)/module/limic.h $(builddir)/module/
	$(MAKE) -C $(KERNEL_SRC) M=$(abs_top_builddir)/module

all-local: module/limic.ko

install-exec-hook: module/limic.ko
	test -z "$(DESTDIR)$(KERNEL_MOD)" ||\
	    $(MKDIR_P) "$(DESTDIR)$(KERNEL_MOD)"
	$(INSTALL) $(builddir)/module/limic.ko $(DESTDIR)$(KERNEL_MOD)/limic.ko

clean-local:
	$(MAKE) -C $(KERNEL_SRC) M=$(abs_top_builddir)/module clean
endif
