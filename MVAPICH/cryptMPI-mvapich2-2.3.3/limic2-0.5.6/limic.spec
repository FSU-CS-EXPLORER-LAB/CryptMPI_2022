Name:           limic2
Version:        0.5.6
Release:        1%{?dist}
Summary:        Linux kernel module for MPI Intra-node Communication

Group:          System Environment/Libraries
License:        Dual BSD/GPL
URL:            http://sslab.konkuk.ac.kr
Source0:        %{name}-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

Autoreq:        1

%description
LiMIC is the Linux kernel module developed by System Software Laboratory
(SSLab) of Konkuk University. It enables high-performance MPI intra-node
communication over multi-core systems. LiMIC achieves very good
performance since it no longer requires MPI libraries to copy messages
in and out of shared memory regions for intra-node communication.
Rather, LiMIC enables messages to be copied directly from sending
process' memory to the receiver process. Thus, LiMIC eliminates
unnecessary message copies and enhances cache utilization.  LiMIC has
two different designs called LiMIC1(LiMIC 1st Generation) and
LiMIC2(LiMIC 2nd Generation). 


%package            module
Summary:            Kernel module for %{name}
Group:              System Environment/Kernel
Requires:           %{name} = %{version}-%{release}

%description        module
The %{name}-module package contains the kernel module for applications
that use %{name}.


%package            common
Summary:            Common files for %{name}-module
Group:              System Environment/Basic
Requires:           %{name} = %{version}-%{release}
Requires(post):     /usr/lib/lsb/install_initd
Requires(preun):    /usr/lib/lsb/remove_initd

%description        common
The %{name}-common package contains udev and lsb compliant init scripts
for applications that use %{name}.


%package            devel
Summary:            Development files for %{name}
Group:              Development/Libraries
Requires:           %{name} = %{version}-%{release}

%description        devel
The %{name}-devel package contains libraries and header files for
developing applications that use %{name}.


%prep
%setup -q


%build
%configure --disable-static --enable-module
make %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT
find $RPM_BUILD_ROOT -name '*.la' -exec rm -f {} ';'


%clean
rm -rf $RPM_BUILD_ROOT


%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%post common
if [ $1 = 1 ]; then
    /usr/lib/lsb/install_initd /etc/init.d/limic
fi

%preun common
if [ $1 = 0 ]; then
    /etc/init.d/limic stop > /dev/null 2>&1
    /usr/lib/lsb/remove_initd /etc/init.d/limic
fi

%files
%defattr(-,root,root,-)
%doc
%{_libdir}/*.so.*

%files module
%defattr(644,root,root,755)
/lib/modules/%(uname -r)/extra/limic.ko

%files common
%defattr(-,root,root,-)
%doc
%{_sysconfdir}/init.d/*
%{_sysconfdir}/udev/rules.d/*

%files devel
%defattr(-,root,root,-)
%doc
%{_includedir}/*
%{_libdir}/*.so


%changelog
* Mon Jan 16 2013 Devendar Bureddy <bureddy@cse.ohio-state.edu> - 0.5.6
- Updated version number

* Mon Apr 11 2011 Jonathan Perkins <perkinjo@cse.ohio-state.edu> - 0.5.5-1
- Updated version number

* Wed Oct 13 2010 Jonathan Perkins <perkinjo@cse.ohio-state.edu>
- Updated version number

* Tue May 18 2010 Jonathan Perkins <perkinjo@cse.ohio-state.edu>
- Updated version number

* Wed Jul 01 2009 Jonathan Perkins <perkinjo@cse.ohio-state.edu>
- Updated version number

* Wed Apr 08 2009 Jonathan Perkins <perkinjo@cse.ohio-state.edu>
- Intial creation of limic2 spec.

