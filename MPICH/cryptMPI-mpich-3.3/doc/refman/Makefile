ALL: mpiman.pdf
SHELL = /bin/sh

INSTALL         = /usr/bin/install -c
INSTALL_PROGRAM = ${INSTALL}
INSTALL_DATA    = ${INSTALL} -m 644
prefix          = /home/gavahi/CryptMPI_2022/MPICH/cryptMPI-mpich-3.3/Install_mpich
datarootdir     = @datatrootdir@
docdir          = ${datarootdir}/doc/${PACKAGE_TARNAME}
srcdir          = .

VPATH=.:${srcdir}

# Use Bourne shell syntax only (export a=b is not universally supported)
LATEX      = TEXINPUTS=".:$(srcdir):" ; export TEXINPUTS ; latex
BIBTEX     = BIBINPUTS=".:$(srcdir):" ; export BIBINPUTS ; bibtex
DVIPDFM    = TEXINPUTS=".:$(srcdir):" ; export TEXINPUTS ; dvipdfm
LATEX2HTML = latex2html

MPIMANAUXFILES = refman.sty tpage.sty refcmd.tex refmpi.tex
mpiman.dvi: mpiman.tex MPI_Wtime.tex
	-rm -f mpiman.log mpiman.msg
	-$(LATEX) $(srcdir)/mpiman < /dev/null 2>&1 >mpiman.msg
	$(LATEX) $(srcdir)/mpiman < /dev/null 2>&1 >mpiman.msg

# Note that some make programs will not allow comments in the commands for 
# targets
# 	for file in $(MPIMANAUXFILES) ; do \
# 	    if [ ! -s $$file ] ; then \
# 	       if [ -s $(srcdir)/$$file ] ; then \
# 	            cp $(srcdir)/$$file . ; \
#                else \
# 		    echo "Unable to find file $$file" ; \
# 		    exit 1; \
#                fi ;\
# 	    fi ; \
# 	done

mpiman.pdf: mpiman.dvi
	$(DVIPDFM) mpiman

clean:
	rm -f *.dvi *.log *.toc *.aux *.out
distclean: clean

#
# Install these into the various FTP and WWW locations
install: mpiman.pdf
	$(INSTALL_DATA) mpiman.pdf ${docdir}
