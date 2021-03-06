#
# makefile.in - Input makefile for the configure script.
#
#

SHELL = /bin/sh

#### Start of system configuration section. ####

prefix = /usr/local
exec_prefix = ${prefix}

bindir = ${exec_prefix}/bin
libdir = /usr/local/lib
includedir = ${prefix}/include

INSTALLDIRS =\
$(bindir)\
$(libdir)\
$(includedir)

DISTDIR = ~/pub_web

INSTALL = /usr/bin/install -c
INSTALL_PROGRAM = ${INSTALL}
INSTALL_DATA = ${INSTALL} -m 644

#### End of system configuration section. ####

CC = gcc
CFLAGS = -g -O2 -Wall -pedantic -fPIC
LDFLAGS = 
CPPFLAGS = 

AR = ar
ARFLAGS = cru

MLIB = -lm
NNLIB = -lnn
CSALIB = -lcsa

SRC =\
getbound.c\
getnodes.c\
gridaverager.c\
gridbathy.c\
gridmap.c\
gridnodes.c\
gucommon.c\
insertgrid.c\
poly.c\
setbathy.c\
subgrid.c\
xy2ij.c

HEADERS =\
gridaverager.h\
gridmap.h\
gridnodes.h\
gucommon.h\
nan.h\
poly.h\
version.h

LIBOBJECTS =\
gridmap.o\
gridnodes.o\
gucommon.o\
poly.o

PROGRAMS =\
getbound\
getnodes\
insertgrid\
setbathy\
subgrid\
xy2ij\
gridbathy

.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -I. $*.c -o $*.o

all: libgu.a $(PROGRAMS)

libgu.a: $(LIBOBJECTS)
	$(AR) $(ARFLAGS) libgu.a $(LIBOBJECTS)
	chmod go+r libgu.a

standalone: override LDFLAGS+=-static
standalone: $(PROGRAMS)
	strip $(PROGRAMS)

getbound: libgu.a getbound.o
	$(CC) -o $@ getbound.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

getnodes: libgu.a getnodes.o
	$(CC) -o $@ getnodes.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

gridbathy: libgu.a gridaverager.o gridbathy.o
	$(CC) -o $@ gridaverager.o gridbathy.o $(CFLAGS) $(LDFLAGS) $(NNLIB) $(CSALIB) libgu.a $(MLIB)

insertgrid: libgu.a insertgrid.o
	$(CC) -o $@ insertgrid.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

setbathy: libgu.a setbathy.o
	$(CC) -o $@ setbathy.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

subgrid: libgu.a subgrid.o
	$(CC) -o $@ subgrid.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

xy2ij: libgu.a xy2ij.o
	$(CC) -o $@ xy2ij.o $(CFLAGS) $(LDFLAGS) libgu.a $(MLIB)

installdirs:
	$(SHELL) mkinstalldirs $(INSTALLDIRS)

install: all installdirs
	for i in libgu.a; do \
	   $(INSTALL_DATA) $$i $(libdir)/$$i; \
	done

	for i in gridmap.h gucommon.h gridnodes.h; do \
	   $(INSTALL_DATA) $$i $(includedir); \
	done

	for i in $(PROGRAMS); do \
	   fname=`basename $$i`; \
	   $(INSTALL_PROGRAM) $$i $(bindir); \
	done

clean:
	rm -f *.o *.a $(PROGRAMS) *~ \#*\# core
	cd test; make clean

configclean:
	rm -f gridutilsconfig.h makefile config.cache config.status config.log

distclean: clean configclean

indent:
	indent -T FILE -T gridmap -T gridnodes -T gridaverager -T extent -T poly -T subgrid -T NODETYPE -T COORDTYPE -T gridstats $(SRC) $(HEADERS)
	rm -f *~

dist:
	WDIR=`pwd`;\
	DIR=`basename $$WDIR`;\
	cp -r $$WDIR /tmp;\
	cd /tmp/$$DIR;\
	rm -rf `find /tmp/$$DIR -name CVS`;\
	if [ -e makefile ] ; then\
		make distclean;\
	fi;\
	/tmp/"$$DIR"/configure;\
	make standalone;\
	chmod a+r+x $(PROGRAMS);\
	mv -f $(PROGRAMS) $(DISTDIR);\
	make distclean;\
	cd $(DISTDIR);\
	mv -f $$DIR".tar.gz" $$DIR".tar.gz.prev";\
	gzip -f $(PROGRAMS);\
	cd /tmp;\
	tar czvf $$DIR".tar.gz" $$DIR;\
	chmod a+r $$DIR".tar.gz";\
	mv -f $$DIR".tar.gz" $(DISTDIR)/$$DIR".tar.gz";\
	rm -rf $$DIR
