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

#include "rdma_impl.h"
#include "mpiimpl.h"
#include "vbuf.h"
#include "dreg.h"
#include "mpiutil.h"

#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);                                    \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Get_rndv_rput
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAILI_Get_rndv_rput(MPIDI_VC_t *vc, 
                                    MPID_Request * req,
                                    MPIDI_CH3I_MRAILI_Rndv_info_t * rndv,
                				    MPL_IOV *iov)
{
    /* This function will register the local buf, send rdma write to target, and send
     * get_resp_kt as rput finsh. Currently, we assume the local buffer is contiguous,
     * datatype cases will be considered later */
    MPIDI_msg_sz_t nbytes;
    int rail;
    vbuf *v;
    MPIDI_STATE_DECL(MPIDI_STATE_GEN2_RNDV_RPUT);
    MPIDI_FUNC_ENTER(MPIDI_STATE_GEN2_RNDV_RPUT);

    MPIDI_CH3I_MRAIL_Prepare_rndv(vc, req);

    MPIDI_CH3I_MRAIL_REVERT_RPUT(req);

    if (MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol) {
        MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(req, rndv);
    }

    rail = MRAILI_Send_select_rail(vc);

    /* STEP 2: Push RDMA write */
    while ((req->mrail.rndv_buf_off < req->mrail.rndv_buf_sz)
            && MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol) {

        GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
        v->sreq = req;
        
        MPIU_Assert(v != NULL);
        
        nbytes = req->mrail.rndv_buf_sz - req->mrail.rndv_buf_off;
        
        if (nbytes > mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize) {
            nbytes = mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize;
        }
        
        DEBUG_PRINT("[buffer content]: offset %d\n", req->mrail.rndv_buf_off);
        MRAILI_RDMA_Put(vc, v,
                (char *) (req->mrail.rndv_buf) + req->mrail.rndv_buf_off,
                ((dreg_entry *) req->mrail.d_entry)->memhandle[vc->
                mrail.rails[rail].hca_index]->lkey,
                (char *) (req->mrail.remote_addr) +
                req->mrail.rndv_buf_off, 
                req->mrail.rkey[vc->mrail.rails[rail].hca_index],
                nbytes, rail);
        req->mrail.rndv_buf_off += nbytes;
    }

    if (MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol) {
        MPIDI_CH3I_MRAILI_rput_complete(vc, iov, 1, (int *)&nbytes, &v, rail);
        v->sreq = req;
    }

    MPIDI_FUNC_EXIT(MPIDI_STATE_GEN2_RNDV_RPUT);
    return MPI_SUCCESS;
}
