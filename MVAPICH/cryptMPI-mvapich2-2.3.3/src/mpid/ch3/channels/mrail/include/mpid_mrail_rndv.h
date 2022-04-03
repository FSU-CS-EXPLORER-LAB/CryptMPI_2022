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

#ifndef __MPID_MRAIL_RNDV_H_
#define __MPID_MRAIL_RNDV_H_

#include <mpiimpl.h>
#include <mpidimpl.h>

int MPID_MRAIL_RndvSend (
    MPID_Request** sreq_p,
    const void* buf,
    MPI_Aint count,
    MPI_Datatype datatype,
    int dt_contig,
    MPIDI_msg_sz_t data_sz,
    MPI_Aint dt_true_lb,
    int rank,
    int tag,
    MPID_Comm* comm,
    int context_offset);

int MPID_MRAIL_RndvRecv (
    MPIDI_VC_t* vc,
    MPID_Request* rreq);

int MPIDI_CH3_iStartRndvMsg(MPIDI_VC_t * vc,
                            MPID_Request * sreq, 
                            MPIDI_CH3_Pkt_t * rts_pkt);

int MPIDI_CH3_iStartRndvTransfer(MPIDI_VC_t * vc,
                                 MPID_Request * rreq);
#endif /* __MPID_MRAIL_RNDV_H_ */

