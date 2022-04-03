/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
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
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <infiniband/verbs.h>
#include "rdma_impl.h"
#include "vbuf.h"
#include "ibv_param.h"
#include "sysreport.h"
#include "smp_smpi.h"
#include "mv2_utils.h"
#include "upmi.h"
#include <inttypes.h>
#include "mv2_mpit_cvars.h"

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : USE_BLOCKING
      category    : CH3
      type        : boolean
      default     : false
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Setting this parameter enables mvapich2 to use blocking mode progress.
        MPI applications do not take up any CPU when they are waiting for incoming
        messages.

    - name        : USE_SHARED_MEM
      category    : CH3
      type        : boolean
      default     : true
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Use shared memory for intra-node communication.

    - name        : ON_DEMAND_THRESHOLD
      category    : CH3
      type        : int
      default     : 64
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This defines threshold for enabling on-demand connection management
        scheme. When the size of the job is larger than the threshold value, on-demand
        connection management will be used.

    - name        : ENABLE_SHARP
      category    : CH3
      type        : int
      default     : 0
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This enables the hardware-based SHArP collectives.

    - name        : SM_SCHEDULING
      category    : CH3
      type        : string
      default     : "FIXED_MAPPING"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the policy that will be used to assign HCAs to each of
        the processes.

    - name        : SMALL_MSG_RAIL_SHARING_POLICY
      category    : CH3
      type        : string
      default     : "ROUND_ROBIN"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the policy that will be used to assign HCAs to each of
        the processes with small message sizes.

    - name        : MED_MSG_RAIL_SHARING_POLICY
      category    : CH3
      type        : string
      default     : "ROUND_ROBIN"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the policy that will be used to assign HCAs to each of
        the processes with medium message sizes.

    - name        : RAIL_SHARING_POLICY
      category    : CH3
      type        : string
      default     : "FIXED_MAPPING"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the policy that will be used to assign HCAs to each of
        the processes.

    - name        : NUM_PORTS
      category    : CH3
      type        : int
      default     : 1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the number of ports per InfiniBand adapter to be used for
        communication per adapter on an end node.

    - name        : NUM_QP_PER_PORT
      category    : CH3
      type        : int
      default     : 1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter indicates number of queue pairs per port to be used for
        communication on an end node. This is useful in the presence of multiple
        send/recv engines available per port for data transfer.

    - name        : IBA_EAGER_THRESHOLD
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the switch point between eager and rendezvous protocol in
        MVAPICH2. For better performance, the value of
        MPIR_CVAR_MV2_IBA_EAGER_THRESHOLD should be set the same as
        MPIR_CVAR_MV2_VBUF_TOTAL_SIZE.

    - name        : STRIPING_THRESHOLD
      category    : CH3
      type        : int
      default     : 8192
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter specifies the message size above which we begin to
        stripe the message across multiple rails (if present).

    - name        : RAIL_SHARING_MED_MSG_THRESHOLD
      category    : CH3
      type        : int
      default     : 2048
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the threshold for the medium message size beyond which
        medium rail sharing striping will take place.

    - name        : RAIL_SHARING_LARGE_MSG_THRESHOLD
      category    : CH3
      type        : int
      default     : 16384
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This specifies the threshold for the large message size beyond which
        large rail sharing striping will be effective.

    - name        : USE_MCAST
      category    : CH3
      type        : int
      default     : 1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Set this to 1, to enable hardware multicast support in collective
        communication.

    - name        : COALESCE_THRESHOLD
      category    : CH3
      type        : int
      default     : 6
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
         This parameter determines the threshhold for message coalescing.

    - name        : USE_COALESCE
      category    : CH3
      type        : int
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Coalesce multiple small messages into a single message to increase small
        message throughput.

    - name        : RNDV_PROTOCOL
      category    : CH3
      type        : string
      default     : "RPUT"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        The value of this variable can be set to choose different rendezvous
        protocols. RPUT (default RDMA-Write) RGET (RDMA Read based), R3
        (send/recv based).

    - name        : SPIN_COUNT
      category    : CH3
      type        : int
      default     : 5000
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This is the number of the connection manager polls for new control
        messages from UD channel for each interrupt. This may be increased to
        reduce the interrupt overhead when many incoming control messages from 
        UD channel at the same time.

    - name        : DEFAULT_MTU
      category    : CH3
      type        : string
      default     : "IBV_MTU_1024"
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        The internal MTU size. For Gen2, this parameter should be a string
        instead of an integer. Valid values are: IBV_MTU_256, IBV_MTU_512,
        IBV_MTU_1024, IBV_MTU_2048, IBV_MTU_4096.

    - name        : NUM_CQES_PER_POLL
      category    : CH3
      type        : int
      default     : 96
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Maximum number of InfiniBand messages retrieved from the completion
        queue in one attempt.

    - name        : USE_RDMA_CM
      category    : CH3
      type        : int 
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter enables the use of RDMA CM for establishing the
        connections.

    - name        : USE_IWARP_MODE
      category    : CH3
      type        : int
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter enables the library to run in iWARP mode.

    - name        : SUPPORT_DPM
      category    : CH3
      type        : int
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This option enables the dynamic process management interface and
        on-demand connection management.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#if ENABLE_PVAR_MV2 
/* Defining handles for CVARs */
MPI_T_cvar_handle mv2_sm_scheduling_handle = NULL;
MPI_T_cvar_handle mv2_small_msg_rail_sharing_policy_handle = NULL;
MPI_T_cvar_handle mv2_med_msg_rail_sharing_policy_handle = NULL;
MPI_T_cvar_handle mv2_rail_sharing_policy_handle = NULL;
MPI_T_cvar_handle mv2_num_ports_handle = NULL;
MPI_T_cvar_handle mv2_num_qp_per_port_handle = NULL;
MPI_T_cvar_handle mv2_iba_eager_threshold_handle = NULL;
MPI_T_cvar_handle mv2_striping_threshold_handle = NULL;
MPI_T_cvar_handle mv2_rail_sharing_med_msg_threshold_handle = NULL;
MPI_T_cvar_handle mv2_rail_sharing_large_msg_threshold_handle = NULL;
MPI_T_cvar_handle mv2_use_mcast_handle = NULL;
MPI_T_cvar_handle mv2_coalesce_threshold_handle = NULL;
MPI_T_cvar_handle mv2_use_coalesce_handle = NULL;
MPI_T_cvar_handle mv2_rndv_protocol_handle = NULL;
MPI_T_cvar_handle mv2_spin_count_handle = NULL;
MPI_T_cvar_handle mv2_default_mtu_handle = NULL;
MPI_T_cvar_handle mv2_num_cqes_per_poll_handle = NULL;
MPI_T_cvar_handle mv2_use_rdma_cm_handle = NULL;
MPI_T_cvar_handle mv2_use_iwarp_mode_handle = NULL;
MPI_T_cvar_handle mv2_support_dpm_handle = NULL;

void mv2_free_cvar_handles()
{
    if (mv2_sm_scheduling_handle) {
        MPIU_Free(mv2_sm_scheduling_handle);
        mv2_sm_scheduling_handle = NULL;
    }
    if (mv2_small_msg_rail_sharing_policy_handle) {
        MPIU_Free(mv2_small_msg_rail_sharing_policy_handle);
        mv2_small_msg_rail_sharing_policy_handle = NULL;
    }
    if (mv2_med_msg_rail_sharing_policy_handle) {
        MPIU_Free(mv2_med_msg_rail_sharing_policy_handle);
        mv2_med_msg_rail_sharing_policy_handle = NULL;
    }
    if (mv2_rail_sharing_policy_handle) {
        MPIU_Free(mv2_rail_sharing_policy_handle);
        mv2_rail_sharing_policy_handle = NULL;
    }
    if (mv2_num_ports_handle) {
        MPIU_Free(mv2_num_ports_handle);
        mv2_num_ports_handle = NULL;
    }
    if (mv2_num_qp_per_port_handle) {
        MPIU_Free(mv2_num_qp_per_port_handle);
        mv2_num_qp_per_port_handle = NULL;
    }
    if (mv2_iba_eager_threshold_handle) {
        MPIU_Free(mv2_iba_eager_threshold_handle);
        mv2_iba_eager_threshold_handle = NULL;
    }
    if (mv2_striping_threshold_handle) {
        MPIU_Free(mv2_striping_threshold_handle);
        mv2_striping_threshold_handle = NULL;
    }
    if (mv2_rail_sharing_med_msg_threshold_handle) {
        MPIU_Free(mv2_rail_sharing_med_msg_threshold_handle);
        mv2_rail_sharing_med_msg_threshold_handle = NULL;
    }
    if (mv2_rail_sharing_large_msg_threshold_handle) {
        MPIU_Free(mv2_rail_sharing_large_msg_threshold_handle);
        mv2_rail_sharing_large_msg_threshold_handle = NULL;
    }
    if (mv2_use_mcast_handle) {
        MPIU_Free(mv2_use_mcast_handle);
        mv2_use_mcast_handle = NULL;
    }
    if (mv2_coalesce_threshold_handle) {
        MPIU_Free(mv2_coalesce_threshold_handle);
        mv2_coalesce_threshold_handle = NULL;
    }
    if (mv2_use_coalesce_handle) {
        MPIU_Free(mv2_use_coalesce_handle);
        mv2_use_coalesce_handle = NULL;
    }
    if (mv2_rndv_protocol_handle) {
        MPIU_Free(mv2_rndv_protocol_handle);
        mv2_rndv_protocol_handle = NULL;
    }
    if (mv2_spin_count_handle) {
        MPIU_Free(mv2_spin_count_handle);
        mv2_spin_count_handle = NULL;
    }
    if (mv2_default_mtu_handle) {
        MPIU_Free(mv2_default_mtu_handle);
        mv2_default_mtu_handle = NULL;
    }
    if (mv2_num_cqes_per_poll_handle) {
        MPIU_Free(mv2_num_cqes_per_poll_handle);
        mv2_num_cqes_per_poll_handle = NULL;
    }
    if (mv2_use_rdma_cm_handle) {
        MPIU_Free(mv2_use_rdma_cm_handle);
        mv2_use_rdma_cm_handle = NULL;
    }
    if (mv2_use_iwarp_mode_handle) {
        MPIU_Free(mv2_use_iwarp_mode_handle);
        mv2_use_iwarp_mode_handle = NULL;
    }
    if (mv2_support_dpm_handle) {
        MPIU_Free(mv2_support_dpm_handle);
        mv2_support_dpm_handle = NULL;
    }
}

int mv2_read_and_check_cvar (mv2_mpit_cvar_access_t container) 
{
    int mpi_errno = MPI_SUCCESS;
    int count = 0;
    int read_value = 0;
    int value;

    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(container.cvar_index, NULL,
            &(container.cvar_handle), &count);
    if (mpi_errno != MPI_SUCCESS) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(container.cvar_handle, &read_value);
    if (mpi_errno != MPI_SUCCESS) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if ((container.skip_if_default_has_set == 1) &&
        (read_value == container.default_cvar_value)) {
        *(container.skip) = 1;
        mpi_errno = MPI_SUCCESS; // no need just to reinforce success 4 test
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv(container.env_name) != NULL) &&
        (container.check4_associate_env_conflict == 1)) {
        value = atoi(getenv(container.env_name));
        if (value != read_value) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
                    "variable: %s and CVAR: %s which have differenc values. It's a "
                    "conflict. %s\n",
                    container.env_name,
                    container.cvar_name,
                    container.env_conflict_error_msg);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        } else {
            /* Environment variable and CVAR was set and they had same value.
             * Do not set internal variable since we may want to do other checks
             * before setting the internal variable later on.
             * This could also happen because MVAPICH2 read the value of the
             * environment variable into the CVAR in mpich_cvars.c. */
            *(container.skip) = 1;
        }
    }
    /* Check if value for CVAR is valid */
    if((container.check_min == 1) && (read_value < container.min_value)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
                "%s is out of range; valid values > %d. (current value: %d)"
                " %s\n",
                container.cvar_name,
                container.min_value,
                read_value,
                container.boundary_error_msg);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    if((container.check_max == 1) && (read_value > container.max_value)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
                "%s is out of range; valid values =< %d.(current value: %d)"
                " %s\n",
                container.cvar_name,
                container.max_value,
                read_value,
                container.boundary_error_msg);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    *(container.value) = read_value;

fn_fail:
fn_exit:
    MPIU_Free(container.cvar_handle);
    return mpi_errno;
}

int mv2_set_sm_scheduling()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_SM_SCHEDULING, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_sm_scheduling_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char* )MPIU_Malloc (count * sizeof(MPI_CHAR));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_sm_scheduling_handle,
                                        (void*) read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "FIXED_MAPPING", 13) == 0) {
        goto fn_fail;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv("MV2_SM_SCHEDULING") != NULL)) {
        char* value = getenv("MV2_SM_SCHEDULING");
        if (strncmp(read_value, value, 13 ) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
                    "variable: MV2_SM_SCHEDULING and CVAR: MPIR_CVAR_SM_SCHEDULING "
                    "differently. This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /* Choose algorithm based on CVAR */
    rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
    rdma_rail_sharing_policy = rdma_get_rail_sharing_policy(read_value);

fn_fail:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

int mv2_set_small_msg_rail_sharing_policy()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_SMALL_MSG_RAIL_SHARING_POLICY, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_small_msg_rail_sharing_policy_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char* )MPIU_Malloc (count * sizeof(MPI_CHAR));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_small_msg_rail_sharing_policy_handle,
                                        (void*) read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "ROUND_ROBIN", 11) == 0) {
        goto fn_fail;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv("MV2_SMALL_MSG_RAIL_SHARING_POLICY") != NULL)) {
        char* value = getenv("MV2_SMALL_MSG_RAIL_SHARING_POLICY");
        if (strncmp(read_value, value, 11 ) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
                    "variable: MV2_SMALL_MSG_RAIL_SHARING_POLICY and CVAR: "
                    "MPIR_CVAR_SMALL_MSG_RAIL_SHARING_POLICY differently. "
                    "This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
    rdma_small_msg_rail_sharing_policy = 
            rdma_get_rail_sharing_policy(read_value);

fn_fail:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

int mv2_set_med_msg_rail_sharing_policy()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_MED_MSG_RAIL_SHARING_POLICY, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_med_msg_rail_sharing_policy_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char* )MPIU_Malloc (count * sizeof(MPI_CHAR));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_med_msg_rail_sharing_policy_handle,
            (void*) read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "ROUND_ROBIN", 11) == 0) {
        goto fn_fail;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv("MV2_MED_MSG_RAIL_SHARING_POLICY") != NULL)) {
        char* value = getenv("MV2_MED_MSG_RAIL_SHARING_POLICY");
        if (strncmp(read_value, value, 11 ) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment"
                    " variable: MV2_MED_MSG_RAIL_SHARING_POLICY and CVAR: "
                    "MPIR_CVAR_MED_MSG_RAIL_SHARING_POLICY differently. "
                    "This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
    rdma_med_msg_rail_sharing_policy = rdma_get_rail_sharing_policy(read_value);

fn_fail:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

int mv2_set_rail_sharing_policy()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_RAIL_SHARING_POLICY, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_rail_sharing_policy_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char* )MPIU_Malloc (count * sizeof(MPI_CHAR));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_rail_sharing_policy_handle,
            (void*) read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "ROUND_ROBIN", 11) == 0) {
        goto fn_fail;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv("MV2_RAIL_SHARING_POLICY") != NULL)) {
        char* value = getenv("MV2_RAIL_SHARING_POLICY");
        if (strncmp(read_value, value, 11 ) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment"
                    " variable: MV2_RAIL_SHARING_POLICY and CVAR: "
                    "MPIR_CVAR_RAIL_SHARING_POLICY differently. "
                    "This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
    rdma_rail_sharing_policy = rdma_med_msg_rail_sharing_policy =
        rdma_small_msg_rail_sharing_policy =
        rdma_get_rail_sharing_policy(read_value);

fn_fail:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

int mv2_set_num_ports()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_NUM_PORTS, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_NUM_PORTS";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_num_ports_handle;
    wrapper.default_cvar_value = DEFAULT_NUM_PORTS;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_NUM_PORTS";
    wrapper.env_conflict_error_msg = "CVAR will be set to default!";
    wrapper.check_max = 1;
    wrapper.max_value = MAX_NUM_PORTS;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_num_ports = read_value;
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_num_qp_per_port()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting= 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_NUM_QP_PER_PORT, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_NUM_QP_PER_PORT";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_num_qp_per_port_handle;
    wrapper.default_cvar_value = DEFAULT_NUM_QP_PER_PORT;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_NUM_QP_PER_PORT";
    wrapper.env_conflict_error_msg = "CVAR will be set to default!";
    wrapper.check_max = 1;
    wrapper.max_value = MAX_NUM_QP_PER_PORT;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_num_qp_per_port = read_value;
#ifdef _ENABLE_UD_
        if ((rdma_num_qp_per_port != 1) && (rdma_enable_only_ud || rdma_enable_hybrid)) {
            int my_rank = -1;
            UPMI_GET_RANK(&my_rank);
            rdma_num_qp_per_port = 1;
            PRINT_INFO((my_rank==0), "Cannot have more than one QP with UD_ONLY / Hybrid mode.\n");
            PRINT_INFO((my_rank==0), "Resetting MV2_NUM_QP_PER_PORT to 1.\n");
        }
#endif /* _ENABLE_UD_ */
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_iba_eager_threshold()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    char strtemp[257];

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_IBA_EAGER_THRESHOLD, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_IBA_EAGER_THRESHOLD";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_iba_eager_threshold_handle;
    wrapper.default_cvar_value = -1;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_IBA_EAGER_THRESHOLD";
    wrapper.env_conflict_error_msg = "CVAR will be set to default!";
    wrapper.check_max = 0;
    wrapper.max_value = -1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if(!skip_setting) {
        sprintf(strtemp, "%d", read_value);
        rdma_iba_eager_threshold =
            user_val_to_bytes(strtemp, "MPIR_CVAR_IBA_EAGER_THRESHOLD");
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_striping_threshold()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    char strtemp [257];
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_STRIPING_THRESHOLD, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_STRIPING_THRESHOLD";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_striping_threshold_handle;
    wrapper.default_cvar_value = STRIPING_THRESHOLD;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_STRIPING_THRESHOLD";
    wrapper.env_conflict_error_msg = "CVAR will be set to default!";
    wrapper.check_max = 0;
    wrapper.max_value = 0;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if(!skip_setting){
        sprintf(strtemp, "%d", read_value);
        striping_threshold = user_val_to_bytes(strtemp, 
                "MPIR_CVAR_STRIPING_THRESHOLD");
        if (striping_threshold <= 0) {
            /* Invalid value - set to computed value */
            striping_threshold =
                rdma_vbuf_total_size * rdma_num_ports * rdma_num_qp_per_port *
                rdma_num_hcas;
        }
        if (striping_threshold < rdma_iba_eager_threshold) {
            /* checking to make sure that the striping threshold is not less
             * than the RNDV threshold since it won't work as expected
             */
            striping_threshold = rdma_iba_eager_threshold;
        }
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_rail_sharing_med_msg_threshold()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    char strtemp [257];

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_RAIL_SHARING_MED_MSG_THRESHOLD, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_RAIL_SHARING_MED_MSG_THRESHOLD";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_rail_sharing_med_msg_threshold_handle;
    wrapper.default_cvar_value = RDMA_DEFAULT_MED_MSG_RAIL_SHARING_THRESHOLD;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_RAIL_SHARING_MED_MSG_THRESHOLD";
    wrapper.env_conflict_error_msg = "CVAR will be set to default!";
    wrapper.check_max = 0;
    wrapper.max_value = 0;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        sprintf(strtemp, "%d", read_value);
        rdma_med_msg_rail_sharing_threshold =
            user_val_to_bytes(strtemp, "MPIR_CVAR_RAIL_SHARING_MED_MSG_THRESHOLD");
        if (rdma_med_msg_rail_sharing_threshold <= 0) {
            rdma_med_msg_rail_sharing_threshold =
                RDMA_DEFAULT_MED_MSG_RAIL_SHARING_THRESHOLD;
        }
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_rail_sharing_large_msg_threshold()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    char strtemp[257];

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_RAIL_SHARING_LARGE_MSG_THRESHOLD, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_RAIL_SHARING_LARGE_MSG_THRESHOLD";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_rail_sharing_large_msg_threshold_handle;
    wrapper.default_cvar_value = RDMA_DEFAULT_LARGE_MSG_RAIL_SHARING_THRESHOLD;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_RAIL_SHARING_LARGE_MSG_THRESHOLD";
    wrapper.env_conflict_error_msg = "the CVAR will set up to VBUF total size!";
    wrapper.check_max = 0;
    wrapper.max_value = -1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        sprintf(strtemp, "%d", read_value);
        rdma_large_msg_rail_sharing_threshold =
            user_val_to_bytes(strtemp, "MPIR_CVAR_RAIL_SHARING_LARGE_MSG_THRESHOLD");
        if (rdma_large_msg_rail_sharing_threshold <= 0) {
            rdma_large_msg_rail_sharing_threshold = rdma_vbuf_total_size;
        }
    }
fn_fail:
    return mpi_errno;
}

#if defined(_MCST_SUPPORT_)
int mv2_set_use_mcast()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_USE_MCAST, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_USE_MCAST";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_use_mcast_handle;
    wrapper.default_cvar_value = USE_MCAST_DEFAULT_FLAG;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_USE_MCAST";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = 1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        /* Multi-cast is only valid if we are either performing a multi-node job
         * or if SMP only is disabled through one of many methods. */
        MPIDI_CH3I_set_smp_only();
        if (!SMP_ONLY) {
            rdma_enable_mcast = !!read_value;
        } else {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nError setting CVAR:"
                    " Multi-cast is only valid if we are either performing a"
                    " multi-node job or if SMP only is disabled through one"
                    " of many methods.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
fn_fail:
    return mpi_errno;
}
#endif /*_MCST_SUPPORT_*/

int mv2_set_coalesce_threshold()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_COALESCE_THRESHOLD, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_COALESCE_THRESHOLD";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_coalesce_threshold_handle;
    wrapper.default_cvar_value = DEFAULT_COALESCE_THRESHOLD;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_COALESCE_THRESHOLD";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 0;
    wrapper.max_value = -1;
    wrapper.check_min = 1;
    wrapper.min_value = 1;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_coalesce_threshold = read_value;
    }

fn_fail:
    return mpi_errno;
}


int mv2_set_use_coalesce()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_USE_COALESCE, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_USE_COALESCE";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_use_coalesce_handle;
    wrapper.default_cvar_value = DEFAULT_USE_COALESCE;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_USE_COALESCE";
    wrapper.env_conflict_error_msg = "the CVAR will set up to zero";
    wrapper.check_max = 1;
    wrapper.max_value = 1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_use_coalesce = !!read_value;
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_spin_count()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_SPIN_COUNT, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_SPIN_COUNT";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_spin_count_handle;
    wrapper.default_cvar_value = DEFAULT_SPIN_COUNT;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_SPIN_COUNT";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 0;
    wrapper.max_value = -1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_blocking_spin_count_threshold = read_value;
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_num_cqes_per_poll()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    char strtemp[257];
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_NUM_CQES_PER_POLL, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_NUM_CQES_PER_POLL";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_num_cqes_per_poll_handle;
    wrapper.default_cvar_value = RDMA_MAX_CQE_ENTRIES_PER_POLL;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_NUM_CQES_PER_POLL";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 0;
    wrapper.max_value = MAX_NUM_CQES_PER_POLL;
    wrapper.check_min = 1;
    wrapper.min_value = MIN_NUM_CQES_PER_POLL;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        rdma_num_cqes_per_poll = read_value;
        if (rdma_num_cqes_per_poll <= 0 ||
                rdma_num_cqes_per_poll >= RDMA_MAX_CQE_ENTRIES_PER_POLL) {
            rdma_num_cqes_per_poll = RDMA_MAX_CQE_ENTRIES_PER_POLL;
        }
    }
fn_fail:
    return mpi_errno;
}

#if defined(RDMA_CM)
int mv2_set_use_rdma_cm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_USE_RDMA_CM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_USE_RDMA_CM";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_use_rdma_cm_handle;
    wrapper.default_cvar_value = 0;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_USE_RDMA_CM";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = 1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm = !!read_value;
    }
fn_fail:
    return mpi_errno;

}

int mv2_set_use_iwarp_mode()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_USE_IWARP_MODE, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_USE_IWARP_MODE";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_use_iwarp_mode_handle;
    wrapper.default_cvar_value = 0;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_USE_IWARP_MODE";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = 1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm = !!(read_value);
        mv2_MPIDI_CH3I_RDMA_Process.use_iwarp_mode = !!(read_value);
    }
fn_fail:
    return mpi_errno;

}
#endif /*RDMA_CM*/

int mv2_set_support_dpm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_SUPPORT_DPM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_SUPPORT_DPM";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_support_dpm_handle;
    wrapper.default_cvar_value = 0;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_SUPPORT_DPM";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = 1;
    wrapper.check_min = 1;
    wrapper.min_value = 0;
    wrapper.boundary_error_msg = NULL;
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
#if defined(RDMA_CM)
        mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm = 0;
        mv2_MPIDI_CH3I_RDMA_Process.use_iwarp_mode = 0;
#endif /*RDMA_CM*/
        mv2_use_eager_fast_send = 0;
        mv2_on_demand_ud_info_exchange = 0;    /* Trac #780 */
        MPIDI_CH3I_Process.has_dpm = read_value;
    }
fn_fail:
    return mpi_errno;
}

int mv2_set_rndv_protocol()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_RNDV_PROTOCOL, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_rndv_protocol_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char*) MPIU_Malloc ( count * sizeof(char));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_rndv_protocol_handle,
            read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "RPUT", 4)==0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (getenv("MV2_RNDV_PROTOCOL") != NULL) {
        char* value = getenv("MV2_RNDV_PROTOCOL");
        if (strncmp(read_value, value, 4) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
                    "variable: MV2_RNDV_PROTOCOL and CVAR: MPIR_CVAR_RNDV_PROTOCOL "
                    "differently. This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /* Choose algorithm based on CVAR */
    if (strncmp(read_value, "RGET", 4) == 0
#ifdef _ENABLE_XRC_
            && !USE_XRC
#endif
       ) {
#if defined(CKPT)
        MPL_usage_printf("MV2_RNDV_PROTOCOL "
                "must be either \"RPUT\" or \"R3\" when checkpoint is enabled\n");
        rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
#else /* defined(CKPT) */
        rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
#endif /* defined(CKPT) */
    } else if (strncmp(read_value, "R3", 2) == 0) {
        rdma_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
    } else {
#ifdef _ENABLE_XRC_
        if (!USE_XRC)
#endif
            /* catching invalid inputs */
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "MPIR_CVAR_RNDV_PROTOCOL "
                    "must be either \"RPUT\", \"RGET\", or \"R3\"\n"
                    "the CVAR will be set to default (\"RPUT\"\n");
        rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
    }
fn_fail:
fn_exit:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

int mv2_set_default_mtu()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    char* read_value = NULL;
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_DEFAULT_MTU, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
            &mv2_default_mtu_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    read_value = (char* )MPIU_Malloc (count * sizeof(MPI_CHAR));
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_default_mtu_handle,
            (void*) read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (strncmp(read_value, "IBV_MTU_1024", 12) == 0) {
        goto fn_fail;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if ((getenv("MV2_DEFAULT_MTU") != NULL)) {
        char* value = getenv("MV2_DEFAULT_MTU");
        if (strncmp(read_value, value, 12 ) != 0) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
                    "variable: MV2_DEFAULT_MTU and CVAR: MPIR_CVAR_DEFAULT_MTU "
                    "differently. This is a conflict. Please use one of them.\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /* Choose algorithm based on CVAR */    
    if (strncmp(read_value, "IBV_MTU_256", 11) == 0) {
        rdma_default_mtu = IBV_MTU_256;
    } else if (strncmp(read_value, "IBV_MTU_512", 11) == 0) {
        rdma_default_mtu = IBV_MTU_512;
    } else if (strncmp(read_value, "IBV_MTU_1024", 12) == 0) {
        rdma_default_mtu = IBV_MTU_1024;
    } else if (strncmp(read_value, "IBV_MTU_2048", 12) == 0) {
        rdma_default_mtu = IBV_MTU_2048;
    } else if (strncmp(read_value, "IBV_MTU_4096", 12) == 0) {
        rdma_default_mtu = IBV_MTU_4096;
    } else {
        rdma_default_mtu = IBV_MTU_1024;
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has entered wrong input."
                " DEFAULT_MTU will be IBV_MTU_1024.\n");
    }
fn_fail:
    if(read_value) {
        MPIU_Free(read_value);
    }
    return mpi_errno;
}

void mv2_update_cvars()
{
    mv2_set_force_arch_type();
    mv2_set_default_mtu();
    mv2_set_rndv_protocol();
    mv2_set_support_dpm();
#if defined(RDMA_CM)
    mv2_set_use_iwarp_mode();
    mv2_set_use_rdma_cm();
#endif /*RDMA_CM*/
    mv2_set_num_cqes_per_poll();
    mv2_set_spin_count();
    mv2_set_use_coalesce();
    mv2_set_coalesce_threshold();
#if defined(_MCST_SUPPORT_)
    mv2_set_use_mcast();
#endif /*_MCST_SUPPORT_*/
    mv2_set_rail_sharing_large_msg_threshold();
    mv2_set_rail_sharing_med_msg_threshold();
    mv2_set_striping_threshold();
    mv2_set_iba_eager_threshold();
    mv2_set_num_qp_per_port();
    mv2_set_num_ports();
    mv2_set_rail_sharing_policy();
    mv2_set_med_msg_rail_sharing_policy();
    mv2_set_small_msg_rail_sharing_policy();
    mv2_set_sm_scheduling();
}
#endif /*ENABLE_PVAR_MV2*/
