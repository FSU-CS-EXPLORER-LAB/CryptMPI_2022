/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#ifndef CH3_HWLOC_BIND_H_
#define CH3_HWLOC_BIND_H_

#include <hwloc.h>
#include <dirent.h>

extern int ib_socket_bind;
extern unsigned int mv2_enable_affinity;
#define MV2_MAX_NUM_SOCKETS_PER_NODE    (16)

typedef struct {
    int num_hca;
    int closest[MV2_MAX_NUM_SOCKETS_PER_NODE];
} tab_socket_t;

typedef enum {
    POLICY_BUNCH = 0,
    POLICY_SCATTER,
    POLICY_HYBRID,
} policy_type_t;

typedef enum {
    LEVEL_CORE,
    LEVEL_MULTIPLE_CORES,
    LEVEL_SOCKET,
    LEVEL_NUMANODE,
} level_type_t;

typedef struct {
    hwloc_obj_t obja;
    hwloc_obj_t objb;
    hwloc_obj_t ancestor;
} ancestor_type;

typedef struct {
    hwloc_obj_t obj;
    cpu_set_t cpuset;
    float load;
} obj_attribute_type;

struct MPIDI_PG;

extern policy_type_t mv2_binding_policy;
extern level_type_t mv2_binding_level;
extern int mv2_user_defined_mapping;
extern unsigned int mv2_enable_affinity;
extern unsigned int mv2_enable_leastload;
extern unsigned int mv2_hca_aware_process_mapping;

extern int s_cpu_mapping_line_max;
extern char *s_cpu_mapping;

void map_scatter_load(obj_attribute_type * tree);
void map_bunch_load(obj_attribute_type * tree);
void map_scatter_core(int num_cpus);
void map_scatter_socket(int num_sockets, hwloc_obj_type_t binding_level);
void map_bunch_core(int num_cpus);
void map_bunch_socket(int num_sockets, hwloc_obj_type_t binding_level);
int get_cpu_mapping_hwloc(long N_CPUs_online, hwloc_topology_t topology);
int get_cpu_mapping(long N_CPUs_online);
#if defined(CHANNEL_MRAIL)
int get_ib_socket(struct ibv_device * ibdev);
#endif /* defined(CHANNEL_MRAIL) */
int smpi_setaffinity(int my_local_id);
int MPIDI_CH3I_set_affinity(struct MPIDI_PG * pg, int pg_rank);

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
extern hwloc_topology_t topology;
extern hwloc_topology_t topology_whole;
int smpi_load_hwloc_topology(void);
int smpi_load_hwloc_topology_whole(void);
int smpi_destroy_hwloc_topology(void);
int smpi_unlink_hwloc_topology_file(void);
#else
static hwloc_topology_t topology = NULL;
static hwloc_topology_t topology_whole = NULL;
static inline int smpi_load_hwloc_topology(void)
{
    if (!topology) {
        hwloc_topology_init(&topology);
        hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
        hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);
        hwloc_topology_load(topology);
    }
}
static inline int smpi_load_hwloc_topology_whole(void)
{
    if (!topology_whole) {
        hwloc_topology_init(&topology_whole);
        hwloc_topology_set_flags(topology_whole, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
        hwloc_topology_set_flags(topology_whole, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);
        hwloc_topology_load(topology_whole);
    }    
}
static inline int smpi_destroy_hwloc_topology(void)
{
    if (topology) {
        hwloc_topology_destroy(topology);
    }
    
    if (topology_whole) {
        hwloc_topology_destroy(topology_whole);
    }
}
#endif

#endif /* CH3_HWLOC_BIND_H_ */
