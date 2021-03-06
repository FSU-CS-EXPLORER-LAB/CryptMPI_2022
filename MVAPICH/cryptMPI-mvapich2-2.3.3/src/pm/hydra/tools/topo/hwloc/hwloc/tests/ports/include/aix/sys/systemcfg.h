/*
 * Copyright ¬© 2011 Universit√© Bordeaux
 * See COPYING in top-level directory.
 */

#ifndef HWLOC_PORT_AIX_SYS_SYSTEMCFG_H
#define HWLOC_PORT_AIX_SYS_SYSTEMCFG_H

struct {
  int dcache_size;
  int dcache_asc;
  int dcache_line;
  int icache_size;
  int icache_asc;
  int icache_line;
  int L2_cache_size;
  int L2_cache_asc;
  int cache_attrib;
} _system_configuration;

#define __power_pc() 1
#define __power_4() 1
#define __power_5() 1
#define __power_6() 1
#define __power_7() 1

#endif /* HWLOC_PORT_AIX_SYS_SYSTEMCFG_H */
