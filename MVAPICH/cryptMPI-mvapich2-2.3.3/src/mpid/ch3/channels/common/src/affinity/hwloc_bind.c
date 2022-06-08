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
#include "mpichconf.h"
#include "mpidimpl.h"
#include "mpidi_ch3_impl.h"
#include <mpimem.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include "upmi.h"
#include "mpiutil.h"
#include "hwloc_bind.h"
#if defined(HAVE_LIBIBVERBS)
#include <hwloc/openfabrics-verbs.h>
#endif
#if defined(CHANNEL_MRAIL)
#include "smp_smpi.h"
#include "rdma_impl.h"
#endif /*defined(CHANNEL_MRAIL)*/
#include "mv2_arch_hca_detect.h"
#include "debug_utils.h"

/* CPU Mapping related definitions */

#define CONFIG_FILE "/proc/cpuinfo"
#define MAX_LINE_LENGTH 512
#define MAX_NAME_LENGTH 64
#define HOSTNAME_LENGTH 255
#define FILENAME_LENGTH 512

/* Hybrid mapping related definitions */
#define HYBRID_LINEAR  0
#define HYBRID_COMPACT 1
#define HYBRID_SPREAD  2
#define HYBRID_BUNCH   3
#define HYBRID_SCATTER 4
#define HYBRID_NUMA    5

const char *mv2_cpu_policy_names[] = {"Bunch", "Scatter", "Hybrid"};
const char *mv2_hybrid_policy_names[] = {"Linear", "Compact", "Spread", "Bunch", "Scatter", "NUMA"};

int mv2_hybrid_binding_policy = HYBRID_LINEAR; /* default as linear */
int mv2_pivot_core_id = 0;     /* specify pivot core to start binding MPI ranks */
int mv2_threads_per_proc = 1;  /* there is atleast one thread which is MPI rank */
int num_sockets = 1; /* default */
int num_physical_cores = 0;
int num_pu = 0;
int hw_threads_per_core = 0;
int *mv2_core_map; /* list of core ids achieved after hwloc tree scanning */
int *mv2_core_map_per_numa; /* list of core ids based on NUMA nodes */

int mv2_my_cpu_id = -1;
int mv2_my_sock_id = -1;
int mv2_my_async_cpu_id = -1;
int *local_core_ids = NULL;
int mv2_user_defined_mapping = FALSE;

#ifdef ENABLE_LLNL_SITE_SPECIFIC_OPTIONS
unsigned int mv2_enable_affinity = 0;
#else
unsigned int mv2_enable_affinity = 1;
#endif /*ENABLE_LLNL_SITE_SPECIFIC_OPTIONS*/
unsigned int mv2_enable_leastload = 0;
unsigned int mv2_hca_aware_process_mapping = 1;

typedef enum {
    CPU_FAMILY_NONE = 0,
    CPU_FAMILY_INTEL,
    CPU_FAMILY_AMD,
} cpu_type_t;

int CLOVERTOWN_MODEL = 15;
int HARPERTOWN_MODEL = 23;
int NEHALEM_MODEL = 26;

int ip = 0;
unsigned long *core_mapping = NULL;
int *obj_tree = NULL;

policy_type_t mv2_binding_policy;
level_type_t mv2_binding_level;
hwloc_topology_t topology = NULL;
hwloc_topology_t topology_whole = NULL;

static int INTEL_XEON_DUAL_MAPPING[] = { 0, 1, 0, 1 };

/* ((0,1),(4,5))((2,3),(6,7)) */
static int INTEL_CLOVERTOWN_MAPPING[] = { 0, 0, 1, 1, 0, 0, 1, 1 };

/* legacy ((0,2),(4,6))((1,3),(5,7)) */
static int INTEL_HARPERTOWN_LEG_MAPPING[] = { 0, 1, 0, 1, 0, 1, 0, 1 };

/* common ((0,1),(2,3))((4,5),(6,7)) */
static int INTEL_HARPERTOWN_COM_MAPPING[] = { 0, 0, 0, 0, 1, 1, 1, 1 };

/* legacy (0,2,4,6)(1,3,5,7) with hyperthreading */
static int INTEL_NEHALEM_LEG_MAPPING[] =
    { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };

/* common (0,1,2,3)(4,5,6,7) with hyperthreading */
static int INTEL_NEHALEM_COM_MAPPING[] =
    { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };

static int AMD_OPTERON_DUAL_MAPPING[] = { 0, 0, 1, 1 };
static int AMD_BARCELONA_MAPPING[] = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };

extern int use_hwloc_cpu_binding;

char *s_cpu_mapping = NULL;
static char *custom_cpu_mapping = NULL;
int s_cpu_mapping_line_max = _POSIX2_LINE_MAX;
static int custom_cpu_mapping_line_max = _POSIX2_LINE_MAX;
char *cpu_mapping = NULL;
char *xmlpath = NULL;
char *whole_topology_xml_path = NULL;
int ib_socket_bind = 0;

#if defined(CHANNEL_MRAIL)
int get_ib_socket(struct ibv_device * ibdev)
{
    hwloc_cpuset_t set = NULL;
    hwloc_obj_t osdev = NULL;
    char string[256];
    int retval = 0;

    if (!(set = hwloc_bitmap_alloc())) {
        goto fn_exit;
    }

    if (hwloc_ibv_get_device_cpuset(topology, ibdev, set)) {
        goto fn_exit;
    }

    osdev = hwloc_get_obj_inside_cpuset_by_type(topology, set,
            HWLOC_OBJ_SOCKET, 0);

    if (NULL == osdev) {
        goto fn_exit;
    }

    /*
     * The hwloc object "string" will have the form "Socket#n" so we are
     * looking at the 8th char to detect which socket is.
     */
    hwloc_obj_type_snprintf(string, sizeof(string), osdev, 1);
    retval = osdev->os_index;

fn_exit:
    if (set) {
        hwloc_bitmap_free(set);
    }
    return retval;
}
#endif /* defined(CHANNEL_MRAIL) */

static int first_num_from_str(char **str)
{
    int val = atoi(*str);
    while (isdigit(**str)) {
        (*str)++;
    }
    return val;
}

static inline int compare_float(const float a, const float b)
{
    const float precision = 0.00001;
    if ((a - precision) < b && (a + precision) > b) {
        return 1;
    } else {
        return 0;
    }
}

static int pid_filter(const struct dirent *dir_obj)
{
    int i;
    int length = strlen(dir_obj->d_name);

    for (i = 0; i < length; i++) {
        if (!isdigit(dir_obj->d_name[i])) {
            return 0;
        }
    }
    return 1;
}

static void find_parent(hwloc_obj_t obj, hwloc_obj_type_t type, hwloc_obj_t * parent)
{
    if ((type == HWLOC_OBJ_CORE) || (type == HWLOC_OBJ_SOCKET)
        || (type == HWLOC_OBJ_NODE)) {
        if (obj->parent->type == type) {
            *parent = obj->parent;
            return;
        } else {
            find_parent(obj->parent, type, parent);
        }
    } else {
        return;
    }
}

static void find_leastload_node(obj_attribute_type * tree, hwloc_obj_t original,
                                hwloc_obj_t * result)
{
    int i, j, k, per, ix, depth_nodes, num_nodes, depth_sockets, num_sockets;
    hwloc_obj_t obj, tmp;

    depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
    num_nodes = hwloc_get_nbobjs_by_depth(topology, depth_nodes);

    /* One socket includes multi numanodes. */
    if (original->type == HWLOC_OBJ_SOCKET) {
        depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
        num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);
        per = num_nodes / num_sockets;
        ix = (original->logical_index) * per;
        if (per == 1) {
            *result = tree[depth_nodes * num_nodes + ix].obj;
        } else {
            i = depth_nodes * num_nodes + ix;
            for (k = 0; k < (per - 1); k++) {
                j = i + k + 1;
                i = (tree[i].load > tree[j].load) ? j : i;
            }
            *result = tree[i].obj;
        }
    } else if (original->type == HWLOC_OBJ_MACHINE) {
        tmp = NULL;
        for (k = 0; k < num_nodes; k++) {
            obj = hwloc_get_obj_by_depth(topology, depth_nodes, k);
            if (tmp == NULL) {
                tmp = obj;
            } else {
                i = depth_nodes * num_nodes + tmp->logical_index;
                j = depth_nodes * num_nodes + obj->logical_index;
                if (tree[i].load > tree[j].load)
                    tmp = obj;
            }
        }
        *result = tmp;
    } else {
        *result = NULL;
    }
    return;
}

static void find_leastload_socket(obj_attribute_type * tree, hwloc_obj_t original,
                                  hwloc_obj_t * result)
{
    int i, j, k, per, ix, depth_sockets, num_sockets, depth_nodes, num_nodes;
    hwloc_obj_t obj, tmp;

    depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
    num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);

    /* One numanode includes multi sockets. */
    if (original->type == HWLOC_OBJ_NODE) {
        depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
        num_nodes = hwloc_get_nbobjs_by_depth(topology, depth_nodes);
        per = num_sockets / num_nodes;
        ix = (original->logical_index) * per;
        if (per == 1) {
            *result = tree[depth_sockets * num_sockets + ix].obj;
        } else {
            i = depth_sockets * num_sockets + ix;
            for (k = 0; k < (per - 1); k++) {
                j = i + k + 1;
                i = (tree[i].load > tree[j].load) ? j : i;
            }
            *result = tree[i].obj;
        }
    } else if (original->type == HWLOC_OBJ_MACHINE) {
        tmp = NULL;
        for (k = 0; k < num_sockets; k++) {
            obj = hwloc_get_obj_by_depth(topology, depth_sockets, k);
            if (tmp == NULL) {
                tmp = obj;
            } else {
                i = depth_sockets * num_sockets + tmp->logical_index;
                j = depth_sockets * num_sockets + obj->logical_index;
                if (tree[i].load > tree[j].load)
                    tmp = obj;
            }
        }
        *result = tmp;
    } else {
        *result = NULL;
    }
    return;
}

static void find_leastload_core(obj_attribute_type * tree, hwloc_obj_t original,
                                hwloc_obj_t * result)
{
    int i, j, k, per, ix;
    int depth_cores, num_cores, depth_sockets, num_sockets, depth_nodes, num_nodes;

    depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);

    /* Core may have Socket or Numanode as direct parent. */
    if (original->type == HWLOC_OBJ_NODE) {
        depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
        num_nodes = hwloc_get_nbobjs_by_depth(topology, depth_nodes);
        per = num_cores / num_nodes;
        ix = (original->logical_index) * per;
        if (per == 1) {
            *result = tree[depth_cores * num_cores + ix].obj;
        } else {
            i = depth_cores * num_cores + ix;
            for (k = 0; k < (per - 1); k++) {
                j = i + k + 1;
                i = (tree[i].load > tree[j].load) ? j : i;
            }
            *result = tree[i].obj;
        }
    } else if (original->type == HWLOC_OBJ_SOCKET) {
        depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
        num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);
        per = num_cores / num_sockets;
        ix = (original->logical_index) * per;
        if (per == 1) {
            *result = tree[depth_cores * num_cores + ix].obj;
        } else {
            i = depth_cores * num_cores + ix;
            for (k = 0; k < (per - 1); k++) {
                j = i + k + 1;
                i = (tree[i].load > tree[j].load) ? j : i;
            }
            *result = tree[i].obj;
        }
    } else {
        *result = NULL;
    }
    return;
}

static void find_leastload_pu(obj_attribute_type * tree, hwloc_obj_t original,
                              hwloc_obj_t * result)
{
    int i, j, k, per, ix, depth_pus, num_pus, depth_cores, num_cores;

    depth_pus = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    num_pus = hwloc_get_nbobjs_by_depth(topology, depth_pus);

    /* Assume: pu only has core as direct parent. */
    if (original->type == HWLOC_OBJ_CORE) {
        depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
        num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);
        per = num_pus / num_cores;
        ix = (original->logical_index) * per;
        if (per == 1) {
            *result = tree[depth_pus * num_pus + ix].obj;
        } else {
            i = depth_pus * num_pus + ix;
            for (k = 0; k < (per - 1); k++) {
                j = i + k + 1;
                i = (tree[i].load > tree[j].load) ? j : i;
            }
            *result = tree[i].obj;
        }
    } else {
        *result = NULL;
    }
    return;
}


static void update_obj_attribute(obj_attribute_type * tree, int ix, hwloc_obj_t obj,
                                 int cpuset, float load)
{
    tree[ix].obj = obj;
    if (!(cpuset < 0)) {
        CPU_SET(cpuset, &(tree[ix].cpuset));
    }
    tree[ix].load += load;
}

static void insert_load(obj_attribute_type * tree, hwloc_obj_t pu, int cpuset, float load)
{
    int k, depth_pus, num_pus = 0;
    int depth_cores, depth_sockets, depth_nodes, num_cores = 0, num_sockets =
        0, num_nodes = 0;
    hwloc_obj_t parent;

    depth_pus = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_PU);
    num_pus = hwloc_get_nbobjs_by_depth(topology, depth_pus);

    depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
    if (depth_nodes != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_nodes = hwloc_get_nbobjs_by_depth(topology, depth_nodes);
    }
    depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
    if (depth_sockets != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);
    }
    depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    if (depth_cores != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);
    }

    /* Add obj, cpuset and load for HWLOC_OBJ_PU */
    k = depth_pus * num_pus + pu->logical_index;
    update_obj_attribute(tree, k, pu, cpuset, load);
    /* Add cpuset and load for HWLOC_OBJ_CORE */
    if (depth_cores != HWLOC_TYPE_DEPTH_UNKNOWN) {
        find_parent(pu, HWLOC_OBJ_CORE, &parent);
        k = depth_cores * num_cores + parent->logical_index;
        update_obj_attribute(tree, k, parent, cpuset, load);
    }
    /* Add cpuset and load for HWLOC_OBJ_SOCKET */
    if (depth_sockets != HWLOC_TYPE_DEPTH_UNKNOWN) {
        find_parent(pu, HWLOC_OBJ_SOCKET, &parent);
        k = depth_sockets * num_sockets + parent->logical_index;
        update_obj_attribute(tree, k, parent, cpuset, load);
    }
    /* Add cpuset and load for HWLOC_OBJ_NODE */
    if (depth_nodes != HWLOC_TYPE_DEPTH_UNKNOWN) {
        find_parent(pu, HWLOC_OBJ_NODE, &parent);
        k = depth_nodes * num_nodes + parent->logical_index;
        update_obj_attribute(tree, k, parent, cpuset, load);
    }
    return;
}

static void cac_load(obj_attribute_type * tree, cpu_set_t cpuset)
{
    int i, j, depth_pus, num_pus;
    float proc_load;
    int num_processes = 0;
    hwloc_obj_t obj;

    depth_pus = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_PU);
    num_pus = hwloc_get_nbobjs_by_depth(topology, depth_pus);

    for (i = 0; i < num_pus; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            num_processes++;
        }
    }

    /* Process is running on num_processes cores; for each core, the load is proc_load. */
    proc_load = 1 / num_processes;

    /*
     * num_objs is HWLOC_OBJ_PU number, and system CPU number;
     * also HWLOC_OBJ_CORE number when HT disabled or without HT.
     */

    for (i = 0; i < num_pus; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            for (j = 0; j < num_pus; j++) {
                obj = hwloc_get_obj_by_depth(topology, depth_pus, j);
                if (obj->os_index == i) {
                    insert_load(tree, obj, i, proc_load);
                }
            }
        }
    }
    return;
}

static void insert_core_mapping(int ix, hwloc_obj_t pu, obj_attribute_type * tree)
{
    core_mapping[ix] = pu->os_index;
    /* This process will be binding to one pu/core.
     * The load for this pu/core is 1; and not update cpuset.
     */
    insert_load(tree, pu, -1, 1);
    return;
}

void map_scatter_load(obj_attribute_type * tree)
{
    int k;
    int depth_cores, depth_sockets, depth_nodes, num_cores = 0;
    hwloc_obj_t root, node, sockets, core_parent, core, result;

    root = hwloc_get_root_obj(topology);

    depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);

    depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);

    depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    if (depth_cores != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);
    }

    k = 0;
    /*Assume: there is always existing SOCKET, but not always existing NUMANODE(like Clovertown). */
    while (k < num_cores) {
        if (depth_nodes == HWLOC_TYPE_DEPTH_UNKNOWN) {
            find_leastload_socket(tree, root, &result);
        } else {
            if ((depth_nodes) < (depth_sockets)) {
                find_leastload_node(tree, root, &result);
                node = result;
                find_leastload_socket(tree, node, &result);
            } else {
                find_leastload_socket(tree, root, &result);
                sockets = result;
                find_leastload_node(tree, sockets, &result);
            }
        }
        core_parent = result;
        find_leastload_core(tree, core_parent, &result);
        core = result;
        find_leastload_pu(tree, core, &result);
        insert_core_mapping(k, result, tree);
        k++;
    }
}

void map_bunch_load(obj_attribute_type * tree)
{
    int i, j, k, per = 0;
    int per_socket_node, depth_pus, num_pus = 0;
    float current_socketornode_load = 0, current_core_load = 0;
    int depth_cores, depth_sockets, depth_nodes, num_cores = 0, num_sockets =
        0, num_nodes = 0;
    hwloc_obj_t root, node, sockets, core_parent, core, pu, result;

    root = hwloc_get_root_obj(topology);

    depth_nodes = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
    if (depth_nodes != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_nodes = hwloc_get_nbobjs_by_depth(topology, depth_nodes);
    }

    depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
    if (depth_sockets != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);
    }

    depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    if (depth_cores != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);
    }

    depth_pus = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    if (depth_pus != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_pus = hwloc_get_nbobjs_by_depth(topology, depth_pus);
    }

    k = 0;
    /*Assume: there is always existing SOCKET, but not always existing NUMANODE(like Clovertown). */
    while (k < num_cores) {
        if (depth_nodes == HWLOC_TYPE_DEPTH_UNKNOWN) {
            find_leastload_socket(tree, root, &result);
            core_parent = result;
            per = num_cores / num_sockets;
            for (i = 0; (i < per) && (k < num_cores); i++) {
                find_leastload_core(tree, core_parent, &result);
                core = result;
                find_leastload_pu(tree, core, &result);
                pu = result;
                if (i == 0) {
                    current_core_load =
                        tree[depth_pus * num_pus + pu->logical_index].load;
                    insert_core_mapping(k, pu, tree);
                    k++;
                } else {
                    if (compare_float
                        (tree[depth_pus * num_pus + pu->logical_index].load,
                         current_core_load)) {
                        insert_core_mapping(k, pu, tree);
                        k++;
                    }
                }
            }
        } else {
            if ((depth_nodes) < (depth_sockets)) {
                find_leastload_node(tree, root, &result);
                node = result;
                per_socket_node = num_sockets / num_nodes;
                for (j = 0; (j < per_socket_node) && (k < num_cores); j++) {
                    find_leastload_socket(tree, node, &result);
                    sockets = result;
                    if (j == 0) {
                        current_socketornode_load =
                            tree[depth_sockets * num_sockets +
                                 sockets->logical_index].load;
                        per = num_cores / num_sockets;
                        for (i = 0; (i < per) && (k < num_cores); i++) {
                            find_leastload_core(tree, sockets, &result);
                            core = result;
                            find_leastload_pu(tree, core, &result);
                            pu = result;
                            if (i == 0) {
                                current_core_load =
                                    tree[depth_pus * num_pus + pu->logical_index].load;
                                insert_core_mapping(k, pu, tree);
                                k++;
                            } else {
                                if (compare_float
                                    (tree[depth_pus * num_pus + pu->logical_index].load,
                                     current_core_load)) {
                                    insert_core_mapping(k, pu, tree);
                                    k++;
                                }
                            }
                        }
                    } else {
                        if (compare_float
                            (tree
                             [depth_sockets * num_sockets + sockets->logical_index].load,
                             current_socketornode_load)) {
                            for (i = 0; (i < per) && (k < num_cores); i++) {
                                find_leastload_core(tree, sockets, &result);
                                core = result;
                                find_leastload_pu(tree, core, &result);
                                pu = result;
                                if (i == 0) {
                                    current_core_load =
                                        tree[depth_pus * num_pus +
                                             pu->logical_index].load;
                                    insert_core_mapping(k, pu, tree);
                                    k++;
                                } else {
                                    if (compare_float
                                        (tree
                                         [depth_pus * num_pus + pu->logical_index].load,
                                         current_core_load)) {
                                        insert_core_mapping(k, pu, tree);
                                        k++;
                                    }
                                }
                            }

                        }
                    }
                }
            } else {    // depth_nodes > depth_sockets
                find_leastload_socket(tree, root, &result);
                sockets = result;
                per_socket_node = num_nodes / num_sockets;
                for (j = 0; (j < per_socket_node) && (k < num_cores); j++) {
                    find_leastload_node(tree, sockets, &result);
                    node = result;
                    if (j == 0) {
                        current_socketornode_load =
                            tree[depth_nodes * num_nodes + node->logical_index].load;
                        per = num_cores / num_sockets;
                        for (i = 0; (i < per) && (k < num_cores); i++) {
                            find_leastload_core(tree, node, &result);
                            core = result;
                            find_leastload_pu(tree, core, &result);
                            pu = result;
                            if (i == 0) {
                                current_core_load =
                                    tree[depth_pus * num_pus + pu->logical_index].load;
                                insert_core_mapping(k, pu, tree);
                                k++;
                            } else {
                                if (compare_float
                                    (tree[depth_pus * num_pus + pu->logical_index].load,
                                     current_core_load)) {
                                    insert_core_mapping(k, pu, tree);
                                    k++;
                                }
                            }
                        }
                    } else {
                        if (compare_float
                            (tree[depth_nodes * num_nodes + node->logical_index].load,
                             current_socketornode_load)) {
                            for (i = 0; (i < per) && (k < num_cores); i++) {
                                find_leastload_core(tree, node, &result);
                                core = result;
                                find_leastload_pu(tree, core, &result);
                                pu = result;
                                if (i == 0) {
                                    current_core_load =
                                        tree[depth_pus * num_pus +
                                             pu->logical_index].load;
                                    insert_core_mapping(k, pu, tree);
                                    k++;
                                } else {
                                    if (compare_float
                                        (tree
                                         [depth_pus * num_pus + pu->logical_index].load,
                                         current_core_load)) {
                                        insert_core_mapping(k, pu, tree);
                                        k++;
                                    }
                                }
                            }
                        }
                    }
                }
            }   /* depth_nodes > depth_sockets */
        }
    }   /* while */
}

/*
 * Compare two hwloc_obj_t of type HWLOC_OBJ_PU according to sibling_rank, used with qsort
 */
static int cmpproc_smt(const void *a, const void *b)
{
    hwloc_obj_t pa = *(hwloc_obj_t *) a;
    hwloc_obj_t pb = *(hwloc_obj_t *) b;
    return (pa->sibling_rank ==
            pb->sibling_rank) ? pa->os_index - pb->os_index : pa->sibling_rank -
        pb->sibling_rank;
}

static int cmpdepth_smt(const void *a, const void *b)
{
    ancestor_type pa = *(ancestor_type *) a;
    ancestor_type pb = *(ancestor_type *) b;
    if ((pa.ancestor)->depth > (pb.ancestor)->depth) {
        return -1;
    } else if ((pa.ancestor)->depth < (pb.ancestor)->depth) {
        return 1;
    } else {
        return 0;
    }
}

static int cmparity_smt(const void *a, const void *b)
{
    ancestor_type pa = *(ancestor_type *) a;
    ancestor_type pb = *(ancestor_type *) b;
    if ((pa.ancestor)->arity > (pb.ancestor)->arity) {
        return -1;
    } else if ((pa.ancestor)->arity < (pb.ancestor)->arity) {
        return 1;
    } else {
        return 0;
    }
}

static void get_first_obj_bunch(hwloc_obj_t * result)
{
    hwloc_obj_t *objs;
    ancestor_type *array;
    int i, j, k, num_objs, num_ancestors;

    if ((num_objs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU)) <= 0) {
        return;
    }

    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_objs * sizeof(hwloc_obj_t))) == NULL) {
        return;
    }

    for (i = 0; i < num_objs; i++) {
        objs[i] = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
    }

    num_ancestors = num_objs * (num_objs - 1) / 2;

    if ((array =
         (ancestor_type *) MPIU_Malloc(num_ancestors * sizeof(ancestor_type))) == NULL) {
        return;
    }

    k = 0;
    for (i = 0; i < (num_objs - 1); i++) {
        for (j = i + 1; j < num_objs; j++) {
            array[k].obja = objs[i];
            array[k].objb = objs[j];
            array[k].ancestor = hwloc_get_common_ancestor_obj(topology, objs[i], objs[j]);
            k++;
        }
    }

    qsort(array, num_ancestors, sizeof(ancestor_type), cmpdepth_smt);

    for (i = 0; i < (num_ancestors - 1); i++) {
        if ((array[i + 1].ancestor)->depth < (array[i].ancestor)->depth) {
            break;
        }
    }

    qsort(array, (i + 1), sizeof(ancestor_type), cmparity_smt);

    *result = array[0].obja;

    MPIU_Free(objs);
    MPIU_Free(array);
    return;
}

static void get_first_socket_bunch(hwloc_obj_t * result, hwloc_obj_type_t binding_level)
{
    hwloc_obj_t *objs;
    ancestor_type *array;
    int i, j, k, num_objs, num_ancestors;

    if ((num_objs = hwloc_get_nbobjs_by_type(topology, binding_level)) <= 0) {
        return;
    }

    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_objs * sizeof(hwloc_obj_t))) == NULL) {
        return;
    }

    for (i = 0; i < num_objs; i++) {
        objs[i] = hwloc_get_obj_by_type(topology, binding_level, i);
    }

    num_ancestors = num_objs * (num_objs - 1) / 2;

    if ((array =
         (ancestor_type *) MPIU_Malloc(num_ancestors * sizeof(ancestor_type))) == NULL) {
        return;
    }

    k = 0;
    for (i = 0; i < (num_objs - 1); i++) {
        for (j = i + 1; j < num_objs; j++) {
            array[k].obja = objs[i];
            array[k].objb = objs[j];
            array[k].ancestor = hwloc_get_common_ancestor_obj(topology, objs[i], objs[j]);
            k++;
        }
    }

    qsort(array, num_ancestors, sizeof(ancestor_type), cmpdepth_smt);

    for (i = 0; i < (num_ancestors - 1); i++) {
        if ((array[i + 1].ancestor)->depth < (array[i].ancestor)->depth) {
            break;
        }
    }

    if (i < num_ancestors - 1)
        qsort(array, (i + 1), sizeof(ancestor_type), cmparity_smt);

    *result = array[0].obja;

    MPIU_Free(objs);
    MPIU_Free(array);
    return;
}

/*
 * Yields "scatter" affinity scenario in core_mapping.
 */
void map_scatter_core(int num_cpus)
{
    hwloc_obj_t *objs, obj, a;
    unsigned *pdist, maxd;
    int i, j, ix, jp, d, s;

    /* Init and load HWLOC_OBJ_PU objects */
    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_cpus * sizeof(hwloc_obj_t *))) == NULL)
        return;

    obj = NULL;
    i = 0;
    while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PU, obj)) != NULL)
        objs[i++] = obj;
    if (i != num_cpus) {
        MPIU_Free(objs);
        return;
    }

    /* Sort HWLOC_OBJ_PU objects according to sibling_rank */
    qsort(objs, num_cpus, sizeof(hwloc_obj_t *), cmpproc_smt);

    /* Init cumulative distances */
    if ((pdist = (unsigned *) MPIU_Malloc(num_cpus * sizeof(unsigned))) == NULL) {
        MPIU_Free(objs);
        return;
    }

    /* Loop over objects, ix is index in objs where sorted objects start */
    ix = num_cpus;
    s = -1;
    while (ix > 0) {
        /* If new group of SMT processors starts, zero distances */
        if (s != objs[0]->sibling_rank) {
            s = objs[0]->sibling_rank;
            for (j = 0; j < ix; j++)
                pdist[j] = 0;
        }
        /*
         * Determine object that has max. distance to all already stored objects.
         * Consider only groups of SMT processors with same sibling_rank.
         */
        maxd = 0;
        jp = 0;
        for (j = 0; j < ix; j++) {
            if ((j) && (objs[j - 1]->sibling_rank != objs[j]->sibling_rank))
                break;
            if (pdist[j] > maxd) {
                maxd = pdist[j];
                jp = j;
            }
        }

        /* Rotate found object to the end of the list, map out found object from distances */
        obj = objs[jp];
        for (j = jp; j < num_cpus - 1; j++) {
            objs[j] = objs[j + 1];
            pdist[j] = pdist[j + 1];
        }
        objs[j] = obj;
        ix--;

        /*
         * Update cumulative distances of all remaining objects with new stored one.
         * If two HWLOC_OBJ_PU objects don't share a common ancestor, the topology is broken.
         * Our scheme cannot be used in this case.
         */
        for (j = 0; j < ix; j++) {
            if ((a = hwloc_get_common_ancestor_obj(topology, obj, objs[j])) == NULL) {
                MPIU_Free(pdist);
                MPIU_Free(objs);
                return;
            }
            d = objs[j]->depth + obj->depth - 2 * a->depth;
            pdist[j] += d * d;
        }
    }

    /* Collect os_indexes into core_mapping */
    for (i = 0; i < num_cpus; i++) {
        core_mapping[i] = objs[i]->os_index;
    }

    MPIU_Free(pdist);
    MPIU_Free(objs);
    return;
}

void map_scatter_socket(int num_sockets, hwloc_obj_type_t binding_level)
{
    hwloc_obj_t *objs, obj, a;
    unsigned *pdist, maxd;
    int i, j, ix, jp, d, s, num_cores;

    /* Init and load HWLOC_OBJ_SOCKET or HWLOC_OBJ_NODE objects */
    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_sockets * sizeof(hwloc_obj_t *))) == NULL)
        return;

    if ((num_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE)) <= 0) {
        return;
    }

    obj = NULL;
    i = 0;
    while ((obj = hwloc_get_next_obj_by_type(topology, binding_level, obj)) != NULL)
        objs[i++] = obj;
    if (i != num_sockets) {
        MPIU_Free(objs);
        return;
    }

    /* Sort HWLOC_OBJ_SOCKET or HWLOC_OBJ_NODE objects according to sibling_rank */
    qsort(objs, num_sockets, sizeof(hwloc_obj_t *), cmpproc_smt);

    /* Init cumulative distances */
    if ((pdist = (unsigned *) MPIU_Malloc(num_sockets * sizeof(unsigned))) == NULL) {
        MPIU_Free(objs);
        return;
    }

    /* Loop over objects, ix is index in objs where sorted objects start */
    ix = num_sockets;
    s = -1;
    while (ix > 0) {
        /* If new group of SMT processors starts, zero distances */
        if (s != objs[0]->sibling_rank) {
            s = objs[0]->sibling_rank;
            for (j = 0; j < ix; j++)
                pdist[j] = 0;
        }
        /*
         * Determine object that has max. distance to all already stored objects.
         * Consider only groups of SMT processors with same sibling_rank.
         */
        maxd = 0;
        jp = 0;
        for (j = 0; j < ix; j++) {
            if ((j) && (objs[j - 1]->sibling_rank != objs[j]->sibling_rank))
                break;
            if (pdist[j] > maxd) {
                maxd = pdist[j];
                jp = j;
            }
        }

        /* Rotate found object to the end of the list, map out found object from distances */
        obj = objs[jp];
        for (j = jp; j < num_sockets - 1; j++) {
            objs[j] = objs[j + 1];
            pdist[j] = pdist[j + 1];
        }
        objs[j] = obj;
        ix--;

        /*
         * Update cumulative distances of all remaining objects with new stored one.
         * If two HWLOC_OBJ_SOCKET or HWLOC_OBJ_NODE objects don't share a common ancestor, the topology is broken.
         * Our scheme cannot be used in this case.
         */
        for (j = 0; j < ix; j++) {
            if ((a = hwloc_get_common_ancestor_obj(topology, obj, objs[j])) == NULL) {
                MPIU_Free(pdist);
                MPIU_Free(objs);
                return;
            }
            d = objs[j]->depth + obj->depth - 2 * a->depth;
            pdist[j] += d * d;
        }
    }

    /* Collect os_indexes into core_mapping */
    for (i = 0, j = 0; i < num_cores; i++, j++) {
        if (j == num_sockets) {
            j = 0;
        }
        core_mapping[i] = hwloc_bitmap_to_ulong((hwloc_const_bitmap_t) (objs[j]->cpuset));
    }

    MPIU_Free(pdist);
    MPIU_Free(objs);
    return;
}

 /*
  * Yields "bunch" affinity scenario in core_mapping.
  */
void map_bunch_core(int num_cpus)
{
    hwloc_obj_t *objs, obj, a;
    unsigned *pdist, mind;
    int i, j, ix, jp, d, s, num_cores, num_pus;

    /* Init and load HWLOC_OBJ_PU objects */
    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_cpus * sizeof(hwloc_obj_t *))) == NULL)
        return;

    obj = NULL;
    i = 0;

    if ((num_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE)) <= 0) {
        MPIU_Free(objs);
        return;
    }

    if ((num_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU)) <= 0) {
        MPIU_Free(objs);
        return;
    }

    /* SMT Disabled */
    if (num_cores == num_pus) {

        get_first_obj_bunch(&obj);

        if (obj == NULL) {
            MPIU_Free(objs);
            return;
        }

        objs[i] = obj;
        i++;

        while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PU, obj)) != NULL) {
            objs[i] = obj;
            i++;
        }

        obj = NULL;
        while (i != num_cpus) {
            obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PU, obj);
            objs[i++] = obj;
        }

        if (i != num_cpus) {
            MPIU_Free(objs);
            return;
        }

    } else {    /* SMT Enabled */

        while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PU, obj)) != NULL)
            objs[i++] = obj;

        if (i != num_cpus) {
            MPIU_Free(objs);
            return;
        }

        /* Sort HWLOC_OBJ_PU objects according to sibling_rank */
        qsort(objs, num_cpus, sizeof(hwloc_obj_t *), cmpproc_smt);
    }

    /* Init cumulative distances */
    if ((pdist = (unsigned *) MPIU_Malloc(num_cpus * sizeof(unsigned))) == NULL) {
        MPIU_Free(objs);
        return;
    }

    /* Loop over objects, ix is index in objs where sorted objects start */
    ix = num_cpus;
    s = -1;
    while (ix > 0) {
        /* If new group of SMT processors starts, zero distances */
        if (s != objs[0]->sibling_rank) {
            s = objs[0]->sibling_rank;
            for (j = 0; j < ix; j++)
                pdist[j] = UINT_MAX;
        }
        /*
         * Determine object that has min. distance to all already stored objects.
         * Consider only groups of SMT processors with same sibling_rank.
         */
        mind = UINT_MAX;
        jp = 0;
        for (j = 0; j < ix; j++) {
            if ((j) && (objs[j - 1]->sibling_rank != objs[j]->sibling_rank))
                break;
            if (pdist[j] < mind) {
                mind = pdist[j];
                jp = j;
            }
        }

        /* Rotate found object to the end of the list, map out found object from distances */
        obj = objs[jp];
        for (j = jp; j < num_cpus - 1; j++) {
            objs[j] = objs[j + 1];
            pdist[j] = pdist[j + 1];
        }
        objs[j] = obj;
        ix--;

        /*
         * Update cumulative distances of all remaining objects with new stored one.
         * If two HWLOC_OBJ_PU objects don't share a common ancestor, the topology is broken.
         * Our scheme cannot be used in this case.
         */
        for (j = 0; j < ix; j++) {
            if ((a = hwloc_get_common_ancestor_obj(topology, obj, objs[j])) == NULL) {
                MPIU_Free(pdist);
                MPIU_Free(objs);
                return;
            }
            d = objs[j]->depth + obj->depth - 2 * a->depth;
            pdist[j] += d * d;
        }
    }

    /* Collect os_indexes into core_mapping */
    for (i = 0; i < num_cpus; i++) {
        core_mapping[i] = objs[i]->os_index;
    }

    MPIU_Free(pdist);
    MPIU_Free(objs);
    return;
}

int check_num_child(hwloc_obj_t obj)
{
    int i = 0, k, num_cores;

    if ((num_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE)) <= 0) {
        return 0;
    }

    for (k = 0; k < num_cores; k++) {
        if (hwloc_bitmap_isset((hwloc_const_bitmap_t) (obj->cpuset), k)) {
            i++;
        }
    }

    return i;
}

void map_bunch_socket(int num_sockets, hwloc_obj_type_t binding_level)
{
    hwloc_obj_t *objs, obj, a;
    unsigned *pdist, mind;
    int i, j, ix, jp, d, s, num_cores, num_pus;

    /* Init and load HWLOC_OBJ_PU objects */
    if ((objs = (hwloc_obj_t *) MPIU_Malloc(num_sockets * sizeof(hwloc_obj_t *))) == NULL)
        return;

    obj = NULL;
    i = 0;

    if ((num_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE)) <= 0) {
        MPIU_Free(objs);
        return;
    }

    if ((num_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU)) <= 0) {
        MPIU_Free(objs);
        return;
    }

    /* SMT Disabled */
    if (num_cores == num_pus) {

        get_first_socket_bunch(&obj, binding_level);

        if (obj == NULL) {
            MPIU_Free(objs);
            return;
        }

        objs[i] = obj;
        i++;

        while ((obj = hwloc_get_next_obj_by_type(topology, binding_level, obj)) != NULL) {
            objs[i] = obj;
            i++;
        }

        obj = NULL;
        while (i != num_sockets) {
            obj = hwloc_get_next_obj_by_type(topology, binding_level, obj);
            objs[i++] = obj;
        }

        if (i != num_sockets) {
            MPIU_Free(objs);
            return;
        }

    } else {    /* SMT Enabled */

        while ((obj = hwloc_get_next_obj_by_type(topology, binding_level, obj)) != NULL)
            objs[i++] = obj;

        if (i != num_sockets) {
            MPIU_Free(objs);
            return;
        }

        /* Sort HWLOC_OBJ_SOCKET or HWLOC_OBJ_NODE objects according to sibling_rank */
        qsort(objs, num_sockets, sizeof(hwloc_obj_t *), cmpproc_smt);

    }

    /* Init cumulative distances */
    if ((pdist = (unsigned *) MPIU_Malloc(num_sockets * sizeof(unsigned))) == NULL) {
        MPIU_Free(objs);
        return;
    }

    /* Loop over objects, ix is index in objs where sorted objects start */
    ix = num_sockets;
    s = -1;
    while (ix > 0) {
        /* If new group of SMT processors starts, zero distances */
        if (s != objs[0]->sibling_rank) {
            s = objs[0]->sibling_rank;
            for (j = 0; j < ix; j++)
                pdist[j] = UINT_MAX;
        }
        /*
         * Determine object that has min. distance to all already stored objects.
         * Consider only groups of SMT processors with same sibling_rank.
         */
        mind = UINT_MAX;
        jp = 0;
        for (j = 0; j < ix; j++) {
            if ((j) && (objs[j - 1]->sibling_rank != objs[j]->sibling_rank))
                break;
            if (pdist[j] < mind) {
                mind = pdist[j];
                jp = j;
            }
        }

        /* Rotate found object to the end of the list, map out found object from distances */
        obj = objs[jp];
        for (j = jp; j < num_sockets - 1; j++) {
            objs[j] = objs[j + 1];
            pdist[j] = pdist[j + 1];
        }
        objs[j] = obj;
        ix--;

        /*
         * Update cumulative distances of all remaining objects with new stored one.
         * If two HWLOC_OBJ_SOCKET or HWLOC_OBJ_NODE objects don't share a common ancestor, the topology is broken.
         * Our scheme cannot be used in this case.
         */
        for (j = 0; j < ix; j++) {
            if ((a = hwloc_get_common_ancestor_obj(topology, obj, objs[j])) == NULL) {
                MPIU_Free(pdist);
                MPIU_Free(objs);
                return;
            }
            d = objs[j]->depth + obj->depth - 2 * a->depth;
            pdist[j] += d * d;
        }
    }

    /* Collect os_indexes into core_mapping */
    int num_child_in_socket[num_sockets];

    for (i = 0; i < num_sockets; i++) {
        num_child_in_socket[i] = check_num_child(objs[i]);
    }

    for (i = 1; i < num_sockets; i++)
        num_child_in_socket[i] += num_child_in_socket[i - 1];

    for (i = 0, j = 0; i < num_cores; i++) {
        if (i == num_child_in_socket[j]) {
            j++;
        }
        core_mapping[i] = hwloc_bitmap_to_ulong((hwloc_const_bitmap_t) (objs[j]->cpuset));
    }

    MPIU_Free(pdist);
    MPIU_Free(objs);
    return;
}

static int num_digits(unsigned long numcpus)
{
    int n_digits = 0;
    while (numcpus > 0) {
        n_digits++;
        numcpus /= 10;
    }
    return n_digits;
}

int get_cpu_mapping_hwloc(long N_CPUs_online, hwloc_topology_t tp)
{
    unsigned topodepth = -1, depth = -1;
    int num_processes = 0, rc = 0, i;
    int num_sockets = 0;
    int num_numanodes = 0;
    int num_cpus = 0;
    char *s;
    struct dirent **namelist;
    pid_t pid;
    obj_attribute_type *tree = NULL;
    char *value;

    /* Determine topology depth */
    topodepth = hwloc_topology_get_depth(tp);
    if (topodepth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        fprintf(stderr, "Warning: %s: Failed to determine topology depth.\n", __func__);
        return (topodepth);
    }

    /* Count number of (logical) processors */
    depth = hwloc_get_type_depth(tp, HWLOC_OBJ_PU);

    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        fprintf(stderr, "Warning: %s: Failed to determine number of processors.\n",
                __func__);
        return (depth);
    }
    if ((num_cpus = hwloc_get_nbobjs_by_type(tp, HWLOC_OBJ_PU)) <= 0) {
        fprintf(stderr, "Warning: %s: Failed to determine number of processors.\n",
                __func__);
        return -1;
    }

    /* Count number of sockets */
    depth = hwloc_get_type_depth(tp, HWLOC_OBJ_SOCKET);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        fprintf(stderr, "Warning: %s: Failed to determine number of sockets.\n",
                __func__);
        return (depth);
    } else {
        num_sockets = hwloc_get_nbobjs_by_depth(tp, depth);
    }

    /* Count number of numanodes */
    depth = hwloc_get_type_depth(tp, HWLOC_OBJ_NODE);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_numanodes = -1;
    } else {
        num_numanodes = hwloc_get_nbobjs_by_depth(tp, depth);
    }

    if (s_cpu_mapping == NULL) {
        /* We need to do allocate memory for the custom_cpu_mapping array
         * and determine the current load on the different cpu's only
         * when the user has not specified a mapping string. If the user
         * has provided a mapping string, it overrides everything.
         */
        /*TODO: might need a better representation as number of cores per node increases */
        unsigned long long_max = ULONG_MAX;
        int n_digits = num_digits(long_max);
        custom_cpu_mapping =
            MPIU_Malloc(sizeof(char) * num_cpus * (n_digits + 1) + 1);
        if (custom_cpu_mapping == NULL) {
            goto error_free;
        }
        MPIU_Memset(custom_cpu_mapping, 0,
                    sizeof(char) * num_cpus * (n_digits + 1) + 1);
        core_mapping = (unsigned long *) MPIU_Malloc(num_cpus * sizeof(unsigned long));
        if (core_mapping == NULL) {
            goto error_free;
        }
        for (i = 0; i < num_cpus; i++) {
            core_mapping[i] = -1;
        }

        tree = MPIU_Malloc(num_cpus * topodepth * sizeof(obj_attribute_type));
        if (tree == NULL) {
            goto error_free;
        }
        for (i = 0; i < num_cpus * topodepth; i++) {
            tree[i].obj = NULL;
            tree[i].load = 0;
            CPU_ZERO(&(tree[i].cpuset));
        }

        if (!(obj_tree = (int *) MPIU_Malloc(num_cpus * topodepth * sizeof(*obj_tree)))) {
            goto error_free;
        }
        for (i = 0; i < num_cpus * topodepth; i++) {
            obj_tree[i] = -1;
        }

        ip = 0;

        /* MV2_ENABLE_LEASTLOAD: map_bunch/scatter or map_bunch/scatter_load */
        if ((value = getenv("MV2_ENABLE_LEASTLOAD")) != NULL) {
            mv2_enable_leastload = atoi(value);
            if (mv2_enable_leastload != 1) {
                mv2_enable_leastload = 0;
            }
        }

        /* MV2_ENABLE_LEASTLOAD=1, map_bunch_load or map_scatter_load is used */
        if (mv2_enable_leastload == 1) {
            /*
             * Get all processes' pid and cpuset.
             * Get numanode, socket, and core current load according to processes running on it.
             */
            num_processes = scandir("/proc", &namelist, pid_filter, alphasort);
            if (num_processes < 0) {
                fprintf(stderr, "Warning: %s: Failed to scandir /proc.\n", __func__);
                return -1;
            } else {
                int status;
                cpu_set_t pid_cpuset;
                CPU_ZERO(&pid_cpuset);

                /* Get cpuset for each running process. */
                for (i = 0; i < num_processes; i++) {
                    pid = atol(namelist[i]->d_name);
                    status = sched_getaffinity(pid, sizeof(pid_cpuset), &pid_cpuset);
                    /* Process completed. */
                    if (status < 0) {
                        continue;
                    }
                    cac_load(tree, pid_cpuset);
                }
                while (num_processes--) {
                    MPIU_Free(namelist[num_processes]);
                }
                MPIU_Free(namelist);
            }

            if (mv2_binding_policy == POLICY_SCATTER) {
                map_scatter_load(tree);
            } else if (mv2_binding_policy == POLICY_BUNCH) {
                map_bunch_load(tree);
            } else {
                goto error_free;
            }
        } else {
            /* MV2_ENABLE_LEASTLOAD != 1 or MV2_ENABLE_LEASTLOAD == NULL, map_bunch or map_scatter is used */
            if (mv2_binding_policy == POLICY_SCATTER) {
                /* Scatter */
                hwloc_obj_type_t binding_level = HWLOC_OBJ_SOCKET;
                if (mv2_binding_level == LEVEL_SOCKET) {
                    map_scatter_socket(num_sockets, binding_level);
                } else if (mv2_binding_level == LEVEL_NUMANODE) {
                    if (num_numanodes == -1) {
                        /* There is not numanode, fallback to socket */
                        map_scatter_socket(num_sockets, binding_level);
                    } else {
                        binding_level = HWLOC_OBJ_NODE;
                        map_scatter_socket(num_numanodes, binding_level);
                    }
                } else {
                    map_scatter_core(num_cpus);
                }

            } else if (mv2_binding_policy == POLICY_BUNCH) {
                /* Bunch */
                hwloc_obj_type_t binding_level = HWLOC_OBJ_SOCKET;
                if (mv2_binding_level == LEVEL_SOCKET) {
                    map_bunch_socket(num_sockets, binding_level);
                } else if (mv2_binding_level == LEVEL_NUMANODE) {
                    if (num_numanodes == -1) {
                        /* There is not numanode, fallback to socket */
                        map_bunch_socket(num_sockets, binding_level);
                    } else {
                        binding_level = HWLOC_OBJ_NODE;
                        map_bunch_socket(num_numanodes, binding_level);
                    }
                } else {
                    map_bunch_core(num_cpus);
                }
            } else {
                goto error_free;
            }
        }

        /* Assemble custom_cpu_mapping string */
        s = custom_cpu_mapping;
        for (i = 0; i < num_cpus; i++) {
            s += sprintf(s, "%lu:", core_mapping[i]);
        }
    }

    /* Done */
    rc = MPI_SUCCESS;

  error_free:
    if (core_mapping != NULL) {
        MPIU_Free(core_mapping);
    }
    if (tree != NULL) {
        MPIU_Free(tree);
    }
    if (obj_tree) {
        MPIU_Free(obj_tree);
    }

    PRINT_DEBUG(DEBUG_INIT_verbose>0,
            "num_cpus: %d, num_sockets: %d, custom_cpu_mapping: %s\n",
            num_cpus, num_sockets, custom_cpu_mapping);

    return rc;
}


int get_cpu_mapping(long N_CPUs_online)
{
    char line[MAX_LINE_LENGTH];
    char input[MAX_NAME_LENGTH];
    char bogus1[MAX_NAME_LENGTH];
    char bogus2[MAX_NAME_LENGTH];
    char bogus3[MAX_NAME_LENGTH];
    int physical_id;            //return value
    int mapping[N_CPUs_online];
    int core_index = 0;
    cpu_type_t cpu_type = 0;
    int model;
    int vendor_set = 0, model_set = 0, num_cpus = 0;

    FILE *fp = fopen(CONFIG_FILE, "r");
    if (fp == NULL) {
        printf("can not open cpuinfo file \n");
        return 0;
    }

    MPIU_Memset(mapping, 0, sizeof(mapping));
    custom_cpu_mapping = (char *) MPIU_Malloc(sizeof(char) * N_CPUs_online * 2);
    if (custom_cpu_mapping == NULL) {
        return 0;
    }
    MPIU_Memset(custom_cpu_mapping, 0, sizeof(char) * N_CPUs_online * 2);

    while (!feof(fp)) {
        MPIU_Memset(line, 0, MAX_LINE_LENGTH);
        fgets(line, MAX_LINE_LENGTH, fp);

        MPIU_Memset(input, 0, MAX_NAME_LENGTH);
        sscanf(line, "%s", input);

        if (!vendor_set) {
            if (strcmp(input, "vendor_id") == 0) {
                MPIU_Memset(input, 0, MAX_NAME_LENGTH);
                sscanf(line, "%s%s%s", bogus1, bogus2, input);

                if (strcmp(input, "AuthenticAMD") == 0) {
                    cpu_type = CPU_FAMILY_AMD;
                } else {
                    cpu_type = CPU_FAMILY_INTEL;
                }
                vendor_set = 1;
            }
        }

        if (!model_set) {
            if (strcmp(input, "model") == 0) {
                sscanf(line, "%s%s%d", bogus1, bogus2, &model);
                model_set = 1;
            }
        }

        if (strcmp(input, "physical") == 0) {
            sscanf(line, "%s%s%s%d", bogus1, bogus2, bogus3, &physical_id);
            mapping[core_index++] = physical_id;
        }
    }

    num_cpus = core_index;
    if (num_cpus == 4) {
        if ((memcmp(INTEL_XEON_DUAL_MAPPING, mapping, sizeof(int) * num_cpus) == 0)
            && (cpu_type == CPU_FAMILY_INTEL)) {
            strcpy(custom_cpu_mapping, "0:2:1:3");
        } else
            if ((memcmp(AMD_OPTERON_DUAL_MAPPING, mapping, sizeof(int) * num_cpus) == 0)
                && (cpu_type == CPU_FAMILY_AMD)) {
            strcpy(custom_cpu_mapping, "0:1:2:3");
        }
    } else if (num_cpus == 8) {
        if (cpu_type == CPU_FAMILY_INTEL) {
            if (model == CLOVERTOWN_MODEL) {
                if (memcmp(INTEL_CLOVERTOWN_MAPPING, mapping, sizeof(int) * num_cpus) ==
                    0) {
                    strcpy(custom_cpu_mapping, "0:1:4:5:2:3:6:7");
                }
            } else if (model == HARPERTOWN_MODEL) {
                if (memcmp(INTEL_HARPERTOWN_LEG_MAPPING, mapping, sizeof(int) * num_cpus)
                    == 0) {
                    strcpy(custom_cpu_mapping, "0:1:4:5:2:3:6:7");
                } else
                    if (memcmp
                        (INTEL_HARPERTOWN_COM_MAPPING, mapping,
                         sizeof(int) * num_cpus) == 0) {
                    strcpy(custom_cpu_mapping, "0:4:2:6:1:5:3:7");
                }
            } else if (model == NEHALEM_MODEL) {
                if (memcmp(INTEL_NEHALEM_LEG_MAPPING, mapping, sizeof(int) * num_cpus) ==
                    0) {
                    strcpy(custom_cpu_mapping, "0:2:4:6:1:3:5:7");
                } else
                    if (memcmp(INTEL_NEHALEM_COM_MAPPING, mapping, sizeof(int) * num_cpus)
                        == 0) {
                    strcpy(custom_cpu_mapping, "0:4:1:5:2:6:3:7");
                }
            }
        }
    } else if (num_cpus == 16) {
        if (cpu_type == CPU_FAMILY_INTEL) {
            if (model == NEHALEM_MODEL) {
                if (memcmp(INTEL_NEHALEM_LEG_MAPPING, mapping, sizeof(int) * num_cpus) ==
                    0) {
                    strcpy(custom_cpu_mapping, "0:2:4:6:1:3:5:7:8:10:12:14:9:11:13:15");
                } else
                    if (memcmp(INTEL_NEHALEM_COM_MAPPING, mapping, sizeof(int) * num_cpus)
                        == 0) {
                    strcpy(custom_cpu_mapping, "0:4:1:5:2:6:3:7:8:12:9:13:10:14:11:15");
                }
            }
        } else if (cpu_type == CPU_FAMILY_AMD) {
            if (memcmp(AMD_BARCELONA_MAPPING, mapping, sizeof(int) * num_cpus) == 0) {
                strcpy(custom_cpu_mapping, "0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15");
            }
        }
    }
    fclose(fp);

    return MPI_SUCCESS;
}

#if defined(CHANNEL_MRAIL)
int get_socket_id (int ib_socket, int cpu_socket, int num_sockets,
        tab_socket_t * tab_socket)
{
    extern int rdma_local_id, rdma_num_hcas;

    int rdma_num_proc_per_hca;
    int offset_id;
    int j;
    int socket_id = ib_socket;
    int delta = cpu_socket / tab_socket[ib_socket].num_hca;

    rdma_num_proc_per_hca = rdma_num_local_procs / rdma_num_hcas;

    if (rdma_num_local_procs % rdma_num_hcas) {
        rdma_num_proc_per_hca++;
    }

    offset_id = rdma_local_id % rdma_num_proc_per_hca;

    if (offset_id < delta) {
        return ib_socket;
    }

    for (j = 0; j < num_sockets - 1; j++) {
        socket_id = tab_socket[ib_socket].closest[j];

        if (tab_socket[socket_id].num_hca == 0) {
            offset_id -= delta;

            if (offset_id < delta) {
                return socket_id;
            }
        }
    }

    /*
     * Couldn't find a free socket, spread remaining processes
     */
    return rdma_local_id % num_sockets;
}

#undef FUNCNAME
#define FUNCNAME mv2_get_cpu_core_closest_to_hca
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_get_cpu_core_closest_to_hca(int my_local_id, int total_num_cores,
                                    int num_sockets, int depth_sockets)
{
    int i = 0, k = 0;
    int ib_hca_selected = 0;
    int selected_socket = 0;
    int cores_per_socket = 0;
    tab_socket_t *tab_socket = NULL;
    int linelen = strlen(custom_cpu_mapping);

    if (linelen < custom_cpu_mapping_line_max) {
        custom_cpu_mapping_line_max = linelen;
    }

    cores_per_socket = total_num_cores / num_sockets;

    /*
     * Make ib_hca_selected global or make this section a function
     */
    if (FIXED_MAPPING == rdma_rail_sharing_policy) {
        ib_hca_selected = rdma_process_binding_rail_offset /
                            rdma_num_rails_per_hca;
    } else {
        ib_hca_selected = 0;
    }

    tab_socket = (tab_socket_t*)MPIU_Malloc(num_sockets * sizeof(tab_socket_t));
    if (NULL == tab_socket) {
        fprintf(stderr, "could not allocate the socket table\n");
        return -1;
    }

    for (i = 0; i < num_sockets; i++) {
        tab_socket[i].num_hca = 0;

        for(k = 0; k < num_sockets; k++) {
            tab_socket[i].closest[k] = -1;
        }
    }

    for (i = 0; i < rdma_num_hcas; i++) {
        struct ibv_device * ibdev = mv2_MPIDI_CH3I_RDMA_Process.ib_dev[i];
        int socket_id = get_ib_socket(ibdev);
        /*
         * Make this information available globally
         */
        if (i == ib_hca_selected) {
            ib_socket_bind = socket_id;
        }
        tab_socket[socket_id].num_hca++;
    }

    hwloc_obj_t obj_src;
    hwloc_obj_t objs[num_sockets];
    char string[20];

    for (i = 0; i < num_sockets; i++) {
        obj_src = hwloc_get_obj_by_type(topology, HWLOC_OBJ_SOCKET,i);
        hwloc_get_closest_objs(topology, obj_src, (hwloc_obj_t *)&objs,
                                num_sockets - 1);

        for (k = 0; k < num_sockets - 1; k++) {
            hwloc_obj_type_snprintf(string, sizeof(string),
                                objs[k], 1);
            tab_socket[i].closest[k] = objs[k]->os_index;
        }
    }

    selected_socket = get_socket_id(ib_socket_bind, cores_per_socket,
                                    num_sockets, tab_socket);
    MPIU_Free(tab_socket);

    return selected_socket;
}
#endif /* defined(CHANNEL_MRAIL) */

#undef FUNCNAME
#define FUNCNAME mv2_get_assigned_cpu_core
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_get_assigned_cpu_core(int my_local_id, char *cpu_mapping, int max_cpu_map_len, char *tp_str)
{
    int i=0, j=0, c=0;
    char *cp = NULL;
    char *tp = cpu_mapping;
    long N_CPUs_online = sysconf(_SC_NPROCESSORS_ONLN);

    while (*tp != '\0') {
        i = 0;
        cp = tp;

        while (*cp != '\0' && *cp != ':' && i < max_cpu_map_len) {
            ++cp;
            ++i;
        }

        if (j == my_local_id) {
            strncpy(tp_str, tp, i);
            c = atoi(tp);
            if ((mv2_binding_level == LEVEL_CORE) && (c < 0 || c >= N_CPUs_online)) {
                fprintf(stderr, "Warning! : Core id %d does not exist on this architecture! \n", c);
                fprintf(stderr, "CPU Affinity is undefined \n");
                mv2_enable_affinity = 0;
                return -1;
            }
            tp_str[i] = '\0';
            return 0;
        }

        if (*cp == '\0') {
            break;
        }

        tp = cp;
        ++tp;
        ++j;
    }

    return -1;
}

#if defined(CHANNEL_MRAIL)
#undef FUNCNAME
#define FUNCNAME smpi_set_progress_thread_affinity
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_set_progress_thread_affinity()
{
    int mpi_errno = MPI_SUCCESS;
    hwloc_cpuset_t cpuset;

    /* Alloc cpuset */
    cpuset = hwloc_bitmap_alloc();
    /* Set cpuset to mv2_my_async_cpu_id */
    hwloc_bitmap_set(cpuset, mv2_my_async_cpu_id);
    /* Attachement progress thread to mv2_my_async_cpu_id */
    hwloc_set_thread_cpubind(topology, pthread_self(), cpuset, 0);
    /* Free cpuset */
    hwloc_bitmap_free(cpuset);

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME smpi_identify_allgather_local_core_ids
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_identify_allgather_local_core_ids(MPIDI_PG_t * pg)
{
    int mpi_errno = MPI_SUCCESS;
    int p = 0;
    MPIDI_VC_t *vc = NULL;
    MPID_Request **request = NULL;
    MPI_Status *status= NULL;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPID_Comm *comm_ptr=NULL;

    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr );

    /* Allocate memory */
    local_core_ids = MPIU_Malloc(g_smpi.num_local_nodes * sizeof(int));
    if (local_core_ids== NULL) {
        ibv_error_abort(GEN_EXIT_ERR, "Failed to allocate memory for local_core_ids\n");
    }
    request = MPIU_Malloc(g_smpi.num_local_nodes * 2 * sizeof(MPID_Request*));
    if (request == NULL) {
        ibv_error_abort(GEN_EXIT_ERR, "Failed to allocate memory for requests\n");
    }
    status = MPIU_Malloc(g_smpi.num_local_nodes * 2 * sizeof(MPI_Status));
    if (request == NULL) {
        ibv_error_abort(GEN_EXIT_ERR, "Failed to allocate memory for statuses\n");
    }
    /* Perform intra-node allgather */
    for (p = 0; p < g_smpi.num_local_nodes; ++p) {
        MPIDI_PG_Get_vc(pg, g_smpi.l2g_rank[p], &vc);
        if (vc->smp.local_nodes >= 0) {
            mpi_errno = MPIC_Irecv((void*)&local_core_ids[vc->smp.local_nodes],
                                    1, MPI_INT, vc->pg_rank, MPIR_ALLGATHER_TAG,
                                    comm_ptr, &request[g_smpi.num_local_nodes+p]);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            mpi_errno = MPIC_Isend((void*)&mv2_my_cpu_id, 1, MPI_INT, vc->pg_rank,
                                    MPIR_ALLGATHER_TAG, comm_ptr, &request[p], &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }
    /* Wait for intra-node allgather to finish */
    mpi_errno = MPIC_Waitall(g_smpi.num_local_nodes*2, request, status, &errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    if (request) {
        MPIU_Free(request);
    }
    if (status) {
        MPIU_Free(status);
    }
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME smpi_identify_free_cores
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_identify_free_cores(hwloc_cpuset_t *sock_cpuset, hwloc_cpuset_t *free_sock_cpuset)
{
    int i = 0;
    int mpi_errno = MPI_SUCCESS;
    int num_sockets = -1;
    int depth_sockets = -1;
    hwloc_obj_t socket = NULL;
    hwloc_cpuset_t my_cpuset = NULL;
    char cpu_str[128];

    /* Alloc cpuset */
    my_cpuset = hwloc_bitmap_alloc();
    *sock_cpuset = hwloc_bitmap_alloc();
    /* Clear CPU set */
    hwloc_bitmap_zero(my_cpuset);
    hwloc_bitmap_zero(*sock_cpuset);
    /* Set cpuset to mv2_my_cpu_id */
    hwloc_bitmap_set(my_cpuset, mv2_my_cpu_id);

    depth_sockets   = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
    num_sockets     = hwloc_get_nbobjs_by_depth(topology, depth_sockets);

    for (i = 0; i < num_sockets; ++i) {
        socket = hwloc_get_obj_by_depth(topology, depth_sockets, i);
        /* Find the list of CPUs we're allowed to use in the socket */
        hwloc_bitmap_and(*sock_cpuset, socket->online_cpuset, socket->allowed_cpuset);
        /* Find the socket the core I'm bound to resides on */
        if (hwloc_bitmap_intersects(my_cpuset, *sock_cpuset)) {
            /* Create a copy to identify list of free coress */
            *free_sock_cpuset = hwloc_bitmap_dup(*sock_cpuset);
            /* Store my sock ID */
            mv2_my_sock_id = i;
            break;
        }
    }
    if (i == num_sockets) {
        mpi_errno = MPI_ERR_OTHER;
        MPIR_ERR_POP(mpi_errno);
    } else {
        /* Remove cores used by processes from list of available cores */
        for (i = 0; i < g_smpi.num_local_nodes; ++i) {
            hwloc_bitmap_clr(*free_sock_cpuset, local_core_ids[i]);
        }
        hwloc_bitmap_snprintf(cpu_str, 128, *free_sock_cpuset);
        PRINT_DEBUG(DEBUG_INIT_verbose, "Free sock_cpuset = %s\n", cpu_str);
    }

    if (my_cpuset) {
        hwloc_bitmap_free(my_cpuset);
    }
fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME smpi_identify_core_for_async_thread
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_identify_core_for_async_thread(MPIDI_PG_t * pg)
{
    int i = 0;
    int mpi_errno = MPI_SUCCESS;
    hwloc_cpuset_t sock_cpuset = NULL;
    hwloc_cpuset_t free_sock_cpuset = NULL;

    /* Gather IDs of cores local processes are bound to */
    mpi_errno = smpi_identify_allgather_local_core_ids(pg);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* Identify my socket and cores available in my socket */
    mpi_errno = smpi_identify_free_cores(&sock_cpuset, &free_sock_cpuset);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* Identify core to be used for async thread */
    if (!hwloc_bitmap_iszero(free_sock_cpuset)) {
        for (i = 0; i < g_smpi.num_local_nodes; ++i) {
            /* If local process 'i' is on a core on my socket */
            if (hwloc_bitmap_isset(sock_cpuset, local_core_ids[i])) {
                mv2_my_async_cpu_id = hwloc_bitmap_next(free_sock_cpuset, mv2_my_async_cpu_id);
                if (i == g_smpi.my_local_id) {
                    break;
                }
            }
        }
        /* Ensure async thread gets bound to a core */
        while (mv2_my_async_cpu_id < 0) {
            mv2_my_async_cpu_id = hwloc_bitmap_next(free_sock_cpuset, mv2_my_async_cpu_id);
        }
    }
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "[local_rank: %d]: sock_id = %d, cpu_id = %d, async_cpu_id = %d\n",
                    g_smpi.my_local_id, mv2_my_sock_id, mv2_my_cpu_id, mv2_my_async_cpu_id);

fn_exit:
    /* Free temporary memory */
    if (local_core_ids) {
        MPIU_Free(local_core_ids);
    }
    /* Free cpuset */
    if (sock_cpuset) {
        hwloc_bitmap_free(sock_cpuset);
    }
    if (free_sock_cpuset) {
        hwloc_bitmap_free(free_sock_cpuset);
    }
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#endif /*defined(CHANNEL_MRAIL)*/

#undef FUNCNAME
#define FUNCNAME SMPI_LOAD_HWLOC_TOPOLOGY_WHOLE
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/* This function is the same as smpi_load_hwloc_topology,
 * but has the HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM set. This is
 * useful for certain launchers/clusters where processes don't 
 * have a whole view of the system (like in the case of jsrun). 
 * It's declared separately to avoid unnecessary overheads in
 * smpi_load_hwloc_topology in cases where a full view of the
 * system is not required. 
 * */
int smpi_load_hwloc_topology_whole(void)
{
    int bcast_topology = 1;
    int mpi_errno = MPI_SUCCESS;
    char *kvsname, *value;
    char *hostname = NULL;
    char *tmppath = NULL;
    int uid, my_local_id;

    MPIDI_STATE_DECL(SMPI_LOAD_HWLOC_TOPOLOGY_WHOLE);
    MPIDI_FUNC_ENTER(SMPI_LOAD_HWLOC_TOPOLOGY_WHOLE);

    if (topology_whole != NULL) {
        goto fn_exit;
    }

    mpi_errno = hwloc_topology_init(&topology_whole);
    hwloc_topology_set_flags(topology_whole,
            HWLOC_TOPOLOGY_FLAG_IO_DEVICES   |
            HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM |
            HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);

    uid = getuid();
    my_local_id = MPIDI_Process.my_pg->ch.local_process_id;
    MPIDI_PG_GetConnKVSname(&kvsname);
 
    if ((value = getenv("MV2_BCAST_HWLOC_TOPOLOGY")) != NULL) {
        bcast_topology = !!atoi(value);
    }

    if (my_local_id < 0) {
        if (MPIDI_Process.my_pg_rank == 0) {
            PRINT_ERROR("WARNING! Invalid my_local_id: %d, Disabling hwloc topology broadcast\n", my_local_id);
        }
        bcast_topology = 0;
    }

    if (!bcast_topology) {
        /* Each process loads topology individually */
        mpi_errno = hwloc_topology_load(topology_whole);
        goto fn_exit;
    }

    hostname = (char *) MPIU_Malloc(sizeof(char) * HOSTNAME_LENGTH);
    tmppath = (char *) MPIU_Malloc(sizeof(char) * FILENAME_LENGTH);
    whole_topology_xml_path = (char *) MPIU_Malloc(sizeof(char) * FILENAME_LENGTH);
    if (hostname == NULL || tmppath == NULL || whole_topology_xml_path == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s", "mv2_hwloc_topology_file");
    }

    if (gethostname(hostname, sizeof (char) * HOSTNAME_LENGTH) < 0) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
                                  "gethostname", strerror(errno));
    }
    sprintf(tmppath, "/tmp/mv2-hwloc-%s-%s-%d-whole.tmp", kvsname, hostname, uid);
    sprintf(whole_topology_xml_path, "/tmp/mv2-hwloc-%s-%s-%d-whole.xml", kvsname, hostname, uid);

    /* Local Rank 0 broadcasts topology using xml */
    if (0 == my_local_id) {
        mpi_errno = hwloc_topology_load(topology_whole);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        mpi_errno = hwloc_topology_export_xml(topology_whole, tmppath);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        if(rename(tmppath, whole_topology_xml_path) < 0) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
                                  "rename", strerror(errno));
        }
    } else {
        while(access(whole_topology_xml_path, F_OK) == -1) {
            usleep(1000);
        }
        mpi_errno = hwloc_topology_set_xml(topology_whole, whole_topology_xml_path);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        mpi_errno = hwloc_topology_load(topology_whole);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    if (hostname) {
        MPIU_Free(hostname);
    }
    if (tmppath) {
        MPIU_Free(tmppath);
    }
    MPIDI_FUNC_EXIT(SMPI_LOAD_HWLOC_TOPOLOGY_WHOLE);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME SMPI_LOAD_HWLOC_TOPOLOGY
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_load_hwloc_topology(void)
{
    int bcast_topology = 1;
    int mpi_errno = MPI_SUCCESS;
    char *kvsname, *value;
    char *hostname = NULL;
    char *tmppath = NULL;
    int uid, my_local_id;

    MPIDI_STATE_DECL(SMPI_LOAD_HWLOC_TOPOLOGY);
    MPIDI_FUNC_ENTER(SMPI_LOAD_HWLOC_TOPOLOGY);

    if (topology != NULL) {
        goto fn_exit;
    }

    mpi_errno = hwloc_topology_init(&topology);
    hwloc_topology_set_flags(topology,
            HWLOC_TOPOLOGY_FLAG_IO_DEVICES   |
    
    /* removing HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM flag since we now 
     * have cpu_cores in the heterogeneity detection logic
     */
     //     HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM |
     
            HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);

    uid = getuid();
    my_local_id = MPIDI_Process.my_pg->ch.local_process_id;
    MPIDI_PG_GetConnKVSname(&kvsname);
 
    if ((value = getenv("MV2_BCAST_HWLOC_TOPOLOGY")) != NULL) {
        bcast_topology = !!atoi(value);
    }

    if (my_local_id < 0) {
        if (MPIDI_Process.my_pg_rank == 0) {
            PRINT_ERROR("WARNING! Invalid my_local_id: %d, Disabling hwloc topology broadcast\n", my_local_id);
        }
        bcast_topology = 0;
    }

    if (!bcast_topology) {
        /* Each process loads topology individually */
        mpi_errno = hwloc_topology_load(topology);
        goto fn_exit;
    }

    hostname = (char *) MPIU_Malloc(sizeof(char) * HOSTNAME_LENGTH);
    tmppath = (char *) MPIU_Malloc(sizeof(char) * FILENAME_LENGTH);
    xmlpath = (char *) MPIU_Malloc(sizeof(char) * FILENAME_LENGTH);
    if (hostname == NULL || tmppath == NULL || xmlpath == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s", "mv2_hwloc_topology_file");
    }

    if (gethostname(hostname, sizeof (char) * HOSTNAME_LENGTH) < 0) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
                                  "gethostname", strerror(errno));
    }
    sprintf(tmppath, "/tmp/mv2-hwloc-%s-%s-%d.tmp", kvsname, hostname, uid);
    sprintf(xmlpath, "/tmp/mv2-hwloc-%s-%s-%d.xml", kvsname, hostname, uid);

    /* Local Rank 0 broadcasts topology using xml */
    if (0 == my_local_id) {
        mpi_errno = hwloc_topology_load(topology);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        mpi_errno = hwloc_topology_export_xml(topology, tmppath);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        if(rename(tmppath, xmlpath) < 0) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
                                  "rename", strerror(errno));
        }
    } else {
        while(access(xmlpath, F_OK) == -1) {
            usleep(1000);
        }
        mpi_errno = hwloc_topology_set_xml(topology, xmlpath);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        mpi_errno = hwloc_topology_load(topology);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    if (hostname) {
        MPIU_Free(hostname);
    }
    if (tmppath) {
        MPIU_Free(tmppath);
    }
    MPIDI_FUNC_EXIT(SMPI_LOAD_HWLOC_TOPOLOGY);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME SMPI_UNLINK_HWLOC_TOPOLOGY_FILE
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_unlink_hwloc_topology_file(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(SMPI_UNLINK_HWLOC_TOPOLOGY_FILE);
    MPIDI_FUNC_ENTER(SMPI_UNLINK_HWLOC_TOPOLOGY_FILE);

    if (xmlpath) {
        unlink(xmlpath);
    }

    if (whole_topology_xml_path) {
        unlink(whole_topology_xml_path);
    }

    MPIDI_FUNC_EXIT(SMPI_UNLINK_HWLOC_TOPOLOGY_FILE);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME SMPI_DESTROY_HWLOC_TOPOLOGY
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_destroy_hwloc_topology(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(SMPI_DESTROY_HWLOC_TOPOLOGY);
    MPIDI_FUNC_ENTER(SMPI_DESTROY_HWLOC_TOPOLOGY);

    if (topology) {
        hwloc_topology_destroy(topology);
        topology = NULL;
    }
    
    if (topology_whole)
    {
        hwloc_topology_destroy(topology_whole);
        topology_whole = NULL;
    }

    if (xmlpath) {
        MPIU_Free(xmlpath);
    }

    if (whole_topology_xml_path) {
        MPIU_Free(whole_topology_xml_path);
    }

    MPIDI_FUNC_EXIT(SMPI_DESTROY_HWLOC_TOPOLOGY);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME smpi_setaffinity
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int smpi_setaffinity(int my_local_id)
{
    int selected_socket = 0;
    int mpi_errno = MPI_SUCCESS;

    hwloc_cpuset_t cpuset;
    MPIDI_STATE_DECL(MPID_STATE_SMPI_SETAFFINITY);
    MPIDI_FUNC_ENTER(MPID_STATE_SMPI_SETAFFINITY);

#if !defined(CHANNEL_MRAIL)
    mv2_hca_aware_process_mapping = 0;
#endif

    PRINT_DEBUG(DEBUG_INIT_verbose>0, 
            "my_local_id: %d, mv2_enable_affinity: %d, mv2_binding_level: %d, mv2_binding_policy: %d\n",
            my_local_id, mv2_enable_affinity, mv2_binding_level, mv2_binding_policy);

    if (mv2_enable_affinity > 0) {
        long N_CPUs_online = sysconf(_SC_NPROCESSORS_ONLN);

        if (N_CPUs_online < 1) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno,
                                      MPI_ERR_OTHER,
                                      "**fail", "%s: %s", "sysconf",
                                      strerror(errno));
        }

        mpi_errno = smpi_load_hwloc_topology();
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
        cpuset = hwloc_bitmap_alloc();

        /* Call the cpu_mapping function to find out about how the
         * processors are numbered on the different sockets.
         * The hardware information gathered from this function
         * is required to determine the best set of intra-node thresholds.
         * However, since the user has specified a mapping pattern,
         * we are not going to use any of our proposed binding patterns
         */
        mpi_errno = get_cpu_mapping_hwloc(N_CPUs_online, topology);
        if (mpi_errno != MPI_SUCCESS) {
            /* In case, we get an error from the hwloc mapping function */
            mpi_errno = get_cpu_mapping(N_CPUs_online);
        }

        if (s_cpu_mapping) {
            /* If the user has specified how to map the processes, use it */
            char tp_str[s_cpu_mapping_line_max + 1];

            mpi_errno = mv2_get_assigned_cpu_core(my_local_id, s_cpu_mapping,
                                                    s_cpu_mapping_line_max, tp_str);
            if (mpi_errno != 0) {
                fprintf(stderr, "Error parsing CPU mapping string\n");
                mv2_enable_affinity = 0;
                MPIU_Free(s_cpu_mapping);
                s_cpu_mapping = NULL;
                goto fn_fail;
            }

            // parsing of the string
            char *token = tp_str;
            int cpunum = 0;
            while (*token != '\0') {
                if (isdigit(*token)) {
                    cpunum = first_num_from_str(&token);
                    if (cpunum >= N_CPUs_online) {
                        fprintf(stderr,
                                "Warning! : Core id %d does not exist on this architecture! \n",
                                cpunum);
                        fprintf(stderr, "CPU Affinity is undefined \n");
                        mv2_enable_affinity = 0;
                        MPIU_Free(s_cpu_mapping);
                        goto fn_fail;
                    }
                    hwloc_bitmap_set(cpuset, cpunum);
                    mv2_my_cpu_id = cpunum;
                    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Set mv2_my_cpu_id = %d\n", mv2_my_cpu_id);
                } else if (*token == ',') {
                    token++;
                } else if (*token == '-') {
                    token++;
                    if (!isdigit(*token)) {
                        fprintf(stderr,
                                "Warning! : Core id %c does not exist on this architecture! \n",
                                *token);
                        fprintf(stderr, "CPU Affinity is undefined \n");
                        mv2_enable_affinity = 0;
                        MPIU_Free(s_cpu_mapping);
                        goto fn_fail;
                    } else {
                        int cpuend = first_num_from_str(&token);
                        if (cpuend >= N_CPUs_online || cpuend < cpunum) {
                            fprintf(stderr,
                                    "Warning! : Core id %d does not exist on this architecture! \n",
                                    cpuend);
                            fprintf(stderr, "CPU Affinity is undefined \n");
                            mv2_enable_affinity = 0;
                            MPIU_Free(s_cpu_mapping);
                            goto fn_fail;
                        }
                        int cpuval;
                        for (cpuval = cpunum + 1; cpuval <= cpuend; cpuval++)
                            hwloc_bitmap_set(cpuset, cpuval);
                    }
                } else if (*token != '\0') {
                    fprintf(stderr,
                            "Warning! Error parsing the given CPU mask! \n");
                    fprintf(stderr, "CPU Affinity is undefined \n");
                    mv2_enable_affinity = 0;
                    MPIU_Free(s_cpu_mapping);
                    goto fn_fail;
                }
            }
            // then attachement
            hwloc_set_cpubind(topology, cpuset, 0);

            MPIU_Free(s_cpu_mapping);
            s_cpu_mapping = NULL;
        } else {
            /* The user has not specified how to map the processes,
             * use the data available in /proc/cpuinfo file to decide
             * on the best cpu mapping pattern
             */
            if (mpi_errno != MPI_SUCCESS || custom_cpu_mapping == NULL) {
                /* For some reason, we were not able to retrieve the cpu mapping
                 * information. We are falling back on the linear mapping.
                 * This may not deliver the best performace
                 */
                hwloc_bitmap_only(cpuset, my_local_id % N_CPUs_online);
                mv2_my_cpu_id = (my_local_id % N_CPUs_online);
                PRINT_DEBUG(DEBUG_INIT_verbose>0, "Set mv2_my_cpu_id = %d\n", mv2_my_cpu_id);
                hwloc_set_cpubind(topology, cpuset, 0);
            } else {
                /*
                 * We have all the information that we need. We will bind the
                 * processes to the cpu's now
                 */
                char tp_str[custom_cpu_mapping_line_max + 1];

                mpi_errno = mv2_get_assigned_cpu_core(my_local_id, custom_cpu_mapping,
                        custom_cpu_mapping_line_max, tp_str);
                if (mpi_errno != 0) {
                    fprintf(stderr, "Error parsing CPU mapping string\n");
                    mv2_enable_affinity = 0;
                    goto fn_fail;
                }

                int cores_per_socket = 0;
#if defined(CHANNEL_MRAIL)
                if (!SMP_ONLY && !mv2_user_defined_mapping) {
                    char *value = NULL;
                    if ((value = getenv("MV2_HCA_AWARE_PROCESS_MAPPING")) != NULL) {
                        mv2_hca_aware_process_mapping = !!atoi(value);
                    }
                    if (likely(mv2_hca_aware_process_mapping)) {
                        int num_cpus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
                        int depth_sockets = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
                        int num_sockets = hwloc_get_nbobjs_by_depth(topology, depth_sockets);

                        selected_socket = mv2_get_cpu_core_closest_to_hca(my_local_id, num_cpus,
                                num_sockets, depth_sockets);
                        if (selected_socket < 0) {
                            fprintf(stderr, "Error getting closest socket\n");
                            mv2_enable_affinity = 0;
                            goto fn_fail;
                        }
                        cores_per_socket = num_cpus/num_sockets;
                    }
                }
#endif /* defined(CHANNEL_MRAIL) */

                if (mv2_binding_level == LEVEL_CORE) {
                    if (
#if defined(CHANNEL_MRAIL)
                        SMP_ONLY ||
#endif
                        mv2_user_defined_mapping || !mv2_hca_aware_process_mapping
                       )
                    {
                        hwloc_bitmap_only(cpuset, atol(tp_str));
                        mv2_my_cpu_id = atol(tp_str);
                        PRINT_DEBUG(DEBUG_INIT_verbose>0, "Set mv2_my_cpu_id = %d\n", mv2_my_cpu_id);
                    } else {
                        hwloc_bitmap_only(cpuset,
                                (atol(tp_str) % cores_per_socket)
                                + (selected_socket * cores_per_socket));
                        mv2_my_cpu_id = ((atol(tp_str) % cores_per_socket)
                                        + (selected_socket * cores_per_socket));
                        PRINT_DEBUG(DEBUG_INIT_verbose>0, "Set mv2_my_cpu_id = %d\n", mv2_my_cpu_id);
                    }
                } else {
                    if (
#if defined(CHANNEL_MRAIL)
                        SMP_ONLY ||
#endif
                        mv2_user_defined_mapping || !mv2_hca_aware_process_mapping
                        ) {
                        hwloc_bitmap_from_ulong(cpuset, atol(tp_str));
                    } else {
                        hwloc_bitmap_from_ulong(cpuset,
                                (atol(tp_str) % cores_per_socket)
                                + (selected_socket * cores_per_socket));
                    }
                }
                hwloc_set_cpubind(topology, cpuset, 0);
            }

            MPIU_Free(custom_cpu_mapping);
        }
        /* Free cpuset */
        hwloc_bitmap_free(cpuset);
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_SMPI_SETAFFINITY);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
void mv2_show_cpu_affinity(int verbosity)
{
    int i = 0, j = 0, num_cpus = 0, my_rank = 0, pg_size = 0;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    char *buf = NULL;
    cpu_set_t *allproc_cpu_set = NULL;
    MPID_Comm *comm_world = NULL;
    MPIDI_VC_t *vc = NULL;

    comm_world = MPIR_Process.comm_world;
    pg_size = comm_world->local_size;
    my_rank = comm_world->rank;

    allproc_cpu_set = (cpu_set_t *) MPIU_Malloc(sizeof(cpu_set_t) * pg_size);
    CPU_ZERO(&allproc_cpu_set[my_rank]);
    sched_getaffinity(0, sizeof(cpu_set_t), &allproc_cpu_set[my_rank]);

    mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, allproc_cpu_set,
                                    sizeof(cpu_set_t), MPI_BYTE, comm_world, &errflag);
    if (mpi_errno != MPI_SUCCESS) {
        fprintf(stderr, "MPIR_Allgather_impl returned error");
        return;
    }
    if (my_rank == 0) {
        char *value;
        value = getenv("OMP_NUM_THREADS");
        num_cpus = sysconf(_SC_NPROCESSORS_CONF);
        fprintf(stderr, "-------------CPU AFFINITY-------------\n");
        fprintf(stderr, "OMP_NUM_THREADS           : %d\n",(value != NULL) ? atoi(value) : 0);
        fprintf(stderr, "MV2_THREADS_PER_PROCESS   : %d\n",mv2_threads_per_proc);        
        fprintf(stderr, "MV2_CPU_BINDING_POLICY    : %s\n",mv2_cpu_policy_names[mv2_binding_policy]);
        /* hybrid binding policy is only applicable when mv2_binding_policy is hybrid */
        if (mv2_binding_policy ==  POLICY_HYBRID) {
            fprintf(stderr, "MV2_HYBRID_BINDING_POLICY : %s\n",
                              mv2_hybrid_policy_names[mv2_hybrid_binding_policy]);
        }
        fprintf(stderr, "--------------------------------------\n");

        buf = (char *) MPIU_Malloc(sizeof(char) * 6 * num_cpus);
        for (i = 0; i < pg_size; i++) {
            MPIDI_Comm_get_vc(comm_world, i, &vc);
            if (vc->smp.local_rank != -1 || verbosity > 1) {
                MPIU_Memset(buf, 0, sizeof(buf));
                for (j = 0; j < num_cpus; j++) {
                    if (CPU_ISSET(j, &allproc_cpu_set[vc->pg_rank])) {
                        sprintf((char *) (buf + strlen(buf)), "%4d", j);
                    }
                }
                fprintf(stderr, "RANK:%2d  CPU_SET: %s\n", i, buf);
            }
        }
        fprintf(stderr, "-------------------------------------\n");
        MPIU_Free(buf);
    }
    MPIU_Free(allproc_cpu_set);
}
#endif /* defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */

#if defined(CHANNEL_MRAIL)
int mv2_show_hca_affinity(int verbosity)
{
    int pg_size = 0;
    int my_rank = 0;
    int i = 0, j = 0, k = 0;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    struct ibv_device **hcas = NULL;

    char *hca_names = NULL; 
    char *all_hca_names = NULL;
    
    MPIDI_VC_t *vc = NULL;
    MPID_Comm *comm_world = NULL;

    comm_world = MPIR_Process.comm_world;
    pg_size = comm_world->local_size;
    my_rank = comm_world->rank;

    hcas = mv2_MPIDI_CH3I_RDMA_Process.ib_dev;
    
    hca_names = (char *) MPIU_Malloc(MAX_NUM_HCAS * (IBV_SYSFS_NAME_MAX+1) 
                                    * sizeof(char));
    k = 0; 
    for(i=0; i < rdma_num_hcas; i++) {
        if (i > 0) {
            strcat(hca_names, " ");
            strcat(hca_names, hcas[i]->name);
        } else {
            strcpy(hca_names, hcas[i]->name);
        }
        PRINT_DEBUG(DEBUG_INIT_verbose>0, "Adding hcas[%d]->name = %s\n", i, hcas[i]->name);
    }
    strcat(hca_names, ";");

    if(my_rank == 0) {
        all_hca_names = (char *) MPIU_Malloc(strlen(hca_names) * pg_size);
    }

    PRINT_DEBUG(DEBUG_INIT_verbose>0, "hca_names = %s, strlen(hca_names) = %ld\n", hca_names, strlen(hca_names));
    mpi_errno = MPIR_Gather_impl(hca_names, strlen(hca_names), MPI_CHAR, 
                    all_hca_names, strlen(hca_names), MPI_CHAR, 0, 
                    comm_world, &errflag);

    if (mpi_errno != MPI_SUCCESS) {
        fprintf(stderr, "MPIR_Allgather_impl returned error: %d", mpi_errno);
        return mpi_errno;
    }
    if(my_rank == 0 && all_hca_names != NULL) {
        fprintf(stderr, "-------------HCA AFFINITY-------------\n");
        j = 0;
    
        char *buffer = MPIU_Malloc(MAX_NUM_HCAS * (IBV_SYSFS_NAME_MAX+1) * sizeof(char));
        for(i = 0; i < pg_size; i++) {
            MPIDI_Comm_get_vc(comm_world, i, &vc);
            if (vc->smp.local_rank != -1 || verbosity > 1) {
                k = 0;
                MPIU_Memset(buffer, 0, sizeof(buffer)); 
                fprintf(stderr, "Process: %d HCAs: ", i);
                while(all_hca_names[j] != ';') {
                    buffer[k] = all_hca_names[j];
                    j++;
                    k++;
                }
                buffer[k] = '\0';
                j++;
                fprintf(stderr, "%s\n", buffer);
            }
        }
        MPIU_Free(buffer);
        
        fprintf(stderr, "-------------------------------------\n");
        MPIU_Free(all_hca_names);
    }
    MPIU_Free(hca_names);
    return mpi_errno;
}
#endif /* defined(CHANNEL_MRAIL) */


/* helper function to get PU ids of a given socket */
void mv2_get_pu_list_on_socket (hwloc_topology_t topology, hwloc_obj_t obj, 
                    int depth, int *pu_ids, int *idx) {
    int i;
    if (obj->type == HWLOC_OBJ_PU) {
        pu_ids[*idx] = obj->os_index;
       *idx = *idx + 1;
        return;
    }

    for (i = 0; i < obj->arity; i++) {
        mv2_get_pu_list_on_socket (topology, obj->children[i], depth+1, pu_ids, idx);
    }

    return;
}

void get_pu_list_on_numanode (hwloc_topology_t topology, hwloc_obj_t obj, int depth, 
                    int *pu_ids, int *idx) {
    int i;
    if (obj->type == HWLOC_OBJ_PU) {
        pu_ids[*idx] = obj->os_index;
        *idx = *idx + 1;
        return;
    }

    for (i = 0; i < obj->arity; i++) {
        get_pu_list_on_numanode (topology, obj->children[i], depth+1, pu_ids, idx);
    }

    return;
}



#undef FUNCNAME
#define FUNCNAME mv2_generate_implicit_cpu_mapping
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int mv2_generate_implicit_cpu_mapping (int local_procs, int num_app_threads) {
    
    hwloc_obj_t obj;

    int i, j, k, l, curr, count, chunk, size, scanned, step, node_offset, node_base_pu;
    int topodepth, num_physical_cores_per_socket ATTRIBUTE((unused)), num_pu_per_socket;
    int num_numanodes, num_pu_per_numanode;
    char mapping [s_cpu_mapping_line_max];
    
    i = j = k = l = curr = count = chunk = size = scanned = step = node_offset = node_base_pu = 0;
    count = mv2_pivot_core_id;
    
    /* call optimized topology load */
    smpi_load_hwloc_topology ();

    num_sockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_SOCKET);
    num_numanodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);

    num_physical_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    num_pu = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);

    num_physical_cores_per_socket = num_physical_cores / num_sockets;
    num_pu_per_socket = num_pu / num_sockets;
    num_pu_per_numanode = num_pu / num_numanodes;

    topodepth = hwloc_get_type_depth (topology, HWLOC_OBJ_CORE);
    obj = hwloc_get_obj_by_depth (topology, topodepth, 0); /* check on core 0*/

    hw_threads_per_core = hwloc_bitmap_weight (obj->allowed_cpuset);
    
    mv2_core_map = MPIU_Malloc(sizeof(int) * num_pu);
    mv2_core_map_per_numa = MPIU_Malloc(sizeof(int) * num_pu);

    /* generate core map of the system by scanning the hwloc tree and save it 
     *  in mv2_core_map array. All the policies below are core_map aware now */
    topodepth = hwloc_get_type_depth (topology, HWLOC_OBJ_SOCKET);
    for (i = 0; i < num_sockets; i++) {
        obj = hwloc_get_obj_by_depth (topology, topodepth, i);
        mv2_get_pu_list_on_socket (topology, obj, topodepth, mv2_core_map, &scanned);
    } 
    
    size = scanned;
        

    /* generate core map of the system basd on NUMA domains by scanning the hwloc 
     * tree and save it in mv2_core_map_per_numa array. NUMA based policies are now 
     * map-aware */
    scanned = 0;
    for (i = 0; i < num_numanodes; i++) {
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
        get_pu_list_on_numanode (topology, obj, topodepth, mv2_core_map_per_numa, &scanned);
    }

    /* make sure total PUs are same when we scanned the machine w.r.t sockets and NUMA */
    MPIU_Assert(size == scanned);

    if (mv2_hybrid_binding_policy == HYBRID_COMPACT) {
        /* Compact mapping: Bind each MPI rank to a single phyical core, and bind
         * its associated threads to the hardware threads of the same physical core.
         * Use first socket followed by the second socket */
        if (num_app_threads > hw_threads_per_core) {
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "WARNING: COMPACT mapping is "
               "only meant for hardware multi-threaded (hyper-threaded) processors. "
               "We have detected that your processor does not have hyper-threading "
               "enabled. Note that proceeding with this option on current system will cause "
               "over-subscription, hence leading to severe performance degradation. "
               "We recommend using LINEAR or SPREAD policy for this run.\n");
        }
        
        for (i = 0; i < local_procs; i++) {
            curr = count;
            for (k = 0; k < num_app_threads; k++) {
                j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d,", mv2_core_map[curr]);
                curr = (curr + 1) % num_pu;
            }
            mapping [--j] = '\0'; 
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, ":");
            count = (count + hw_threads_per_core) % num_pu;
        }
    } else if (mv2_hybrid_binding_policy == HYBRID_LINEAR) {
        /* Linear mapping: Bind each MPI rank as well as its associated threads to
         * phyical cores. Only use hardware threads when you run out of physical
         * resources  */
        for (i = 0; i < local_procs; i++) {
            for (k = 0; k < num_app_threads; k++) {
                j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d,", mv2_core_map[curr]);

                curr = ((curr + hw_threads_per_core) >= num_pu) ?
                            ((curr + hw_threads_per_core+ ++step) % num_pu) :
                            (curr + hw_threads_per_core) % num_pu;
            }
            mapping [--j] = '\0';
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, ":");
        }    
    } else if (mv2_hybrid_binding_policy == HYBRID_SPREAD) {
        /* Spread mapping: Evenly distributes all the PUs among MPI ranks and
         * ensures that no two MPI ranks get bound to the same phyiscal core. */
        if (num_physical_cores < local_procs) {
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "WARNING: This configuration "
                        "might lead to oversubscription of cores !!!\n");
            /* limit the mapping to max available PUs */
            num_physical_cores = num_pu;
        }
        chunk = num_physical_cores / local_procs;

        if (chunk > 1) {
            for (i = 0; i < local_procs; i++) {
                 for (k = curr; k < curr+chunk; k++) {
                     for (l = 0; l < hw_threads_per_core; l++) {
                        j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d,", 
                                mv2_core_map[k * hw_threads_per_core + l]);
                     }
                 }
                 mapping [--j] = '\0';
                 j += snprintf (mapping+j, _POSIX2_LINE_MAX, ":");
                 curr = (curr + chunk) % size;
            } 
        } else {
            /* when MPI ranks are more than half-subscription but less than full-subcription, 
             * instead of following the bunch strategy, try to spread-out the ranks evenly 
             * across all the PUs available on all the sockets
             */

            int ranks_per_sock = local_procs / num_sockets;

            curr = 0;
            for (i = 0; i < num_sockets; i++) {
                for (k = curr; k < curr+ranks_per_sock; k++) {
                    for (l = 0; l < hw_threads_per_core; l++) {
                        j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d,",
                                mv2_core_map[k * hw_threads_per_core + l]);
                    }
                    mapping [--j] = '\0';
                    j += snprintf (mapping+j, _POSIX2_LINE_MAX, ":");
                }
                curr = (curr + num_pu_per_socket * chunk) % size;
            }
        }
    } else if (mv2_hybrid_binding_policy == HYBRID_BUNCH) {
        /* Bunch mapping: Bind each MPI rank to a single phyical core of first
         * socket followed by second secket */
        for (i = 0; i < local_procs; i++) {
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d:", mv2_core_map[k]);
            k = (k + hw_threads_per_core) % size;
        } 
    } else if (mv2_hybrid_binding_policy == HYBRID_SCATTER) {
        /* scatter mapping: Bind consecutive MPI ranks to different sockets in
         * round-robin fashion */
        if (num_sockets < 2) {
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "WARNING: Scatter is not a valid policy "
                    "for single-socket systems. Please re-run with Bunch or any other "
                    "applicable policy\n");
            return MPI_ERR_OTHER;
        }
        for (i = 0; i < local_procs; i++) {
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d:", mv2_core_map[k]);
            k = (i % num_sockets == 0) ?
                    (k + num_pu_per_socket) % size :
                    (k + num_pu_per_socket + hw_threads_per_core) % size;
        }
    } else if (mv2_hybrid_binding_policy == HYBRID_NUMA) {
        /* NUMA mapping: Bind consecutive MPI ranks to different NUMA domains in
         * round-robin fashion. */
        for (i = 0; i < local_procs; i++) {
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, "%d,", 
                               mv2_core_map_per_numa[node_base_pu+node_offset]);
            mapping [--j] = '\0';
            j += snprintf (mapping+j, _POSIX2_LINE_MAX, ":");
            node_base_pu = (node_base_pu + num_pu_per_numanode) % size;
            node_offset = (node_base_pu == 0) ? 
                            (node_offset + ((hw_threads_per_core > 0) ? hw_threads_per_core : 1)) : 
                            node_offset;
        }
    }

    /* copy the generated mapping string to final mapping*/
    s_cpu_mapping = (char *) MPIU_Malloc (sizeof (char) * j);
    strncpy (s_cpu_mapping, mapping, j);
    s_cpu_mapping[j-1] = '\0';

    if (MPIDI_Process.my_pg_rank == 0) {
        PRINT_DEBUG(DEBUG_INIT_verbose>0, "num_physical_cores_per_socket %d, mapping: %s", 
                num_physical_cores_per_socket, s_cpu_mapping);
    }
    
    /* cleanup */
    MPIU_Free(mv2_core_map);
    MPIU_Free(mv2_core_map_per_numa);
     
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_set_affinity
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_set_affinity(MPIDI_PG_t * pg, int pg_rank)
{
    char *value;
    int mpi_errno = MPI_SUCCESS;
    int my_local_id;
    int num_local_procs;
    long N_CPUs_online;
    mv2_arch_type arch_type;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SET_AFFINITY);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SET_AFFINITY);

    num_local_procs = MPIDI_Num_local_processes (pg);
    
    N_CPUs_online = sysconf(_SC_NPROCESSORS_ONLN);

    if ((value = getenv("MV2_ENABLE_AFFINITY")) != NULL) {
        mv2_enable_affinity = atoi(value);
    }

    arch_type = mv2_get_arch_type ();
    /* set CPU_BINDING_POLICY=hybrid for Power, Skylake, Frontera, and KNL */
    if (arch_type == MV2_ARCH_IBM_POWER8 ||
        arch_type == MV2_ARCH_IBM_POWER9 ||
        arch_type == MV2_ARCH_INTEL_XEON_PHI_7250 ||
        arch_type == MV2_ARCH_INTEL_PLATINUM_8170_2S_52 ||
        arch_type == MV2_ARCH_INTEL_PLATINUM_8160_2S_48 ||
        arch_type == MV2_ARCH_INTEL_PLATINUM_8280_2S_56 || /* frontera */
        arch_type == MV2_ARCH_AMD_EPYC_7551_64 /* EPYC */ ||
        arch_type == MV2_ARCH_AMD_EPYC_7742_128 /* rome */) {
        setenv ("MV2_CPU_BINDING_POLICY", "hybrid", 0);
        
        /* if system is Frontera, further force hybrid_binding_policy to spread */
        if (arch_type == MV2_ARCH_INTEL_PLATINUM_8280_2S_56) {
            setenv ("MV2_HYBRID_BINDING_POLICY", "spread", 0);
        }
        
        /* if CPU is EPYC, further force hybrid_binding_policy to NUMA */
        if (arch_type == MV2_ARCH_AMD_EPYC_7551_64 ||
            arch_type == MV2_ARCH_AMD_EPYC_7742_128 /* rome */) {
            setenv ("MV2_HYBRID_BINDING_POLICY", "numa", 0);
        } 
    }

    if (mv2_enable_affinity && (num_local_procs > N_CPUs_online)) {
        if (MPIDI_Process.my_pg_rank == 0) {
            PRINT_ERROR ("WARNING: You are running %d MPI processes on a processor "
                            "that supports up to %ld cores. If you still wish to run "
                            "in oversubscribed mode, please set MV2_ENABLE_AFFINITY=0 "
                            "and re-run the program.\n\n", 
                            num_local_procs, N_CPUs_online);

            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "MV2_ENABLE_AFFINITY: oversubscribed cores.");
        }
        goto fn_fail;
    }

    if (mv2_enable_affinity && (value = getenv("MV2_CPU_MAPPING")) != NULL) {
        /* Affinity is on and the user has supplied a cpu mapping string */
        int linelen = strlen(value);
        if (linelen < s_cpu_mapping_line_max) {
            s_cpu_mapping_line_max = linelen;
        }
        s_cpu_mapping =
            (char *) MPIU_Malloc(sizeof(char) * (s_cpu_mapping_line_max + 1));
        strncpy(s_cpu_mapping, value, s_cpu_mapping_line_max);
        s_cpu_mapping[s_cpu_mapping_line_max] = '\0';
        mv2_user_defined_mapping = TRUE;
    }

    if (mv2_enable_affinity && (value = getenv("MV2_CPU_MAPPING")) == NULL) {
        /* Affinity is on and the user has not specified a mapping string */
        if ((value = getenv("MV2_CPU_BINDING_POLICY")) != NULL) {
            /* User has specified a binding policy */
            if (!strcmp(value, "bunch") || !strcmp(value, "BUNCH")) {
                mv2_binding_policy = POLICY_BUNCH;
            } else if (!strcmp(value, "scatter") || !strcmp(value, "SCATTER")) {
                mv2_binding_policy = POLICY_SCATTER;
            } else if (!strcmp(value, "hybrid") || !strcmp(value, "HYBRID")) {
                mv2_binding_policy = POLICY_HYBRID;
               /* check if the OMP_NUM_THREADS is exported or user has
                * explicitly set MV2_THREADS_PER_PROCESS variable*/
               if ((value = getenv("OMP_NUM_THREADS")) != NULL) {
                   mv2_threads_per_proc = atoi (value);
                   if (mv2_threads_per_proc < 0) {
                       if (MPIDI_Process.my_pg_rank == 0) {
                           PRINT_ERROR ("OMP_NUM_THREADS: value can not be set to negative.\n");
                           MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                   "**fail", "**fail %s",
                                   "OMP_NUM_THREADS: negative value.");
                       }
                   }
               }

               if ((value = getenv("MV2_THREADS_PER_PROCESS")) != NULL) {
                   mv2_threads_per_proc = atoi (value);
                   if (mv2_threads_per_proc < 0) {
                       if (MPIDI_Process.my_pg_rank == 0) {
                           PRINT_ERROR ("MV2_THREADS_PER_PROCESS: "
                                   "value can not be set to negative.\n");
                           MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                   "**fail", "**fail %s",
                                   "MV2_THREADS_PER_PROCESS: negative value.");
                       }
                   }
               }

               if (mv2_threads_per_proc > 0) {
                   if ( (mv2_threads_per_proc * num_local_procs) > N_CPUs_online) {
                       if (MPIDI_Process.my_pg_rank == 0) {
                           PRINT_ERROR ("User defined values for MV2_CPU_BINDING_POLICY and "
                                   "MV2_THREADS_PER_PROCESS will lead to oversubscription of "
                                   "the available CPUs. If this was intentional, please "
                                   "re-run the application after setting MV2_ENABLE_AFFINITY=0 or "
                                   "with explicit CPU mapping using MV2_CPU_MAPPING.\n"); 
                           MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                   "**fail", "**fail %s",
                                   "CPU_BINDING_PRIMITIVE: over-subscribed hybrid configuration.");
                       }
                   } 
                    
                   /* Check to see if any pivot core is designated */
                   if ((value = getenv("MV2_PIVOT_CORE_ID")) != NULL) {
                       mv2_pivot_core_id = atoi(value);
                   }
                   
                   /* since mv2_threads_per_proc > 0, check if any threads
                    * binding policy have been explicitly specified */
                   if ((value = getenv("MV2_HYBRID_BINDING_POLICY")) != NULL) {
                       if (!strcmp(value, "linear") || !strcmp(value, "LINEAR")) {
                           mv2_hybrid_binding_policy = HYBRID_LINEAR;
                       } else if (!strcmp(value, "compact") || !strcmp(value, "COMPACT")) {
                           mv2_hybrid_binding_policy = HYBRID_COMPACT;
                       } else if (!strcmp(value, "spread") || !strcmp(value, "SPREAD")) {
                           mv2_hybrid_binding_policy = HYBRID_SPREAD;
                       } else if (!strcmp(value, "bunch") || !strcmp(value, "BUNCH")) {
                           mv2_hybrid_binding_policy = HYBRID_BUNCH;
                       } else if (!strcmp(value, "scatter") || !strcmp(value, "SCATTER")) {
                           mv2_hybrid_binding_policy = HYBRID_SCATTER;
                       } else if (!strcmp(value, "numa") || !strcmp(value, "NUMA")) {
                           /* we only force NUMA binding if we have more than 2 ppn,
                            * otherwise we use bunch (linear) mapping */
                           mv2_hybrid_binding_policy =
                               (num_local_procs > 2) ?  HYBRID_NUMA : HYBRID_LINEAR;
                       }
                   }

                   mv2_binding_level = LEVEL_MULTIPLE_CORES;
               
               } else {
                       PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "WARNING: Process mapping "
                               "mode has been set to 'hybrid' "
                               "indicating an attempt to run a multi-threaded program. However, "
                               "neither the MV2_THREADS_PER_PROCESS nor OMP_NUM_THREADS have been "
                               "set. Please set either one of these variable to the number threads "
                               "desired per process for optimal performance\n");
                               
               }
            } else {
                PRINT_INFO((MPIDI_Process.my_pg_rank == 0),
                            "MV2_CPU_BINDING_POLICY should be "
                            "bunch, scatter or hybrid (upper or lower case).\n");
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                            "**fail", "**fail %s",
                            "CPU_BINDING_PRIMITIVE: Policy should be bunch, scatter or hybrid.");
            }
            mv2_user_defined_mapping = TRUE;
        } else {
            /* User has not specified a binding policy.
             * We are going to do "hybrid-bunch" binding, by default  */
            mv2_binding_policy = POLICY_HYBRID;
        }
    }

    /* generate implicit mapping string based on hybrid binding policy */
    if (mv2_binding_policy == POLICY_HYBRID) {
        mpi_errno = mv2_generate_implicit_cpu_mapping (num_local_procs, 
               mv2_threads_per_proc);
        if (mpi_errno != MPI_SUCCESS) {
           goto fn_fail;
        }
    }

    if (mv2_enable_affinity && (value = getenv("MV2_CPU_MAPPING")) == NULL) {
        /* Affinity is on and the user has not specified a mapping string */
        if ((value = getenv("MV2_CPU_BINDING_LEVEL")) != NULL) {
            /* User has specified a binding level */
            if (!strcmp(value, "core") || !strcmp(value, "CORE")) {
                mv2_binding_level = LEVEL_CORE;
            } else if (!strcmp(value, "socket") || !strcmp(value, "SOCKET")) {
                mv2_binding_level = LEVEL_SOCKET;
            } else if (!strcmp(value, "numanode") || !strcmp(value, "NUMANODE")) {
                mv2_binding_level = LEVEL_NUMANODE;
            } else {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "CPU_BINDING_PRIMITIVE: Level should be core, socket, or numanode.");
            }
            if (MV2_ARCH_INTEL_XEON_PHI_7250 == arch_type &&
                    mv2_binding_level != LEVEL_CORE) {
                if (MPIDI_Process.my_pg_rank == 0) {
                    fprintf(stderr, "CPU_BINDING_PRIMITIVE: Only core level binding supported for this architecture.\n");
                }
                mpi_errno = MPI_ERR_OTHER;
                goto fn_fail;
            }
            mv2_user_defined_mapping = TRUE;
        } else {
            /* User has not specified a binding level and we've not
             * assigned LEVEL_MULTIPLE_CORES earlier. We are going to
             * do "core" binding, by default  */
            if (mv2_binding_level != LEVEL_MULTIPLE_CORES) {
                mv2_binding_level = LEVEL_CORE;
            }
        }
    }

    if (mv2_enable_affinity) {
        my_local_id = pg->ch.local_process_id;
        mpi_errno = smpi_setaffinity(my_local_id);
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SET_AFFINITY);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
