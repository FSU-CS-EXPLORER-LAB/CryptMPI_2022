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

#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);  fflush(stderr);                   \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

extern int g_atomics_support;

struct ibv_mr * register_memory(void * buf, size_t len, int hca_num)
{
    struct ibv_mr * mr;

    if (g_atomics_support) {
        mr = ibv_reg_mr(mv2_MPIDI_CH3I_RDMA_Process.ptag[hca_num], buf, len,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
    } else {
        mr = ibv_reg_mr(mv2_MPIDI_CH3I_RDMA_Process.ptag[hca_num], buf, len,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ);
    }
    DEBUG_PRINT("register return mr %p, buf %p, len %d\n", mr, buf, len);
    return mr;
}

int deregister_memory(struct ibv_mr * mr)
{
    int ret;

    ret = ibv_dereg_mr(mr);
    DEBUG_PRINT("deregister mr %p, ret %d\n", mr, ret);
    return ret;
}

/* Rail selection policy */
int MRAILI_Send_select_rail(MPIDI_VC_t * vc)
{
    static int i = 0;

    if (ROUND_ROBIN == rdma_rail_sharing_policy) {
        i = (i + 1) % rdma_num_rails;
        return i;
    } else if (FIXED_MAPPING == rdma_rail_sharing_policy) {
        i = ((i + 1) % rdma_num_rails_per_hca) +
             rdma_process_binding_rail_offset;
        return i;
    } else {
        /* Currently the best scenario for latency */
        return 0;
    }
}

void *mv2_vbuf_ctrl_buf = NULL;
void *mv2_vbuf_rdma_buf = NULL;
dreg_entry *mv2_rdma_fp_dreg = NULL;
int mv2_rdma_fp_prealloc_buf_index = 0;

int mv2_preallocate_rdma_fp_bufs()
{
    int mpi_errno = MPI_SUCCESS;
    int pagesize = getpagesize();

    /* allocate vbuf struct buffers */
    if(MPIU_Memalign((void **) &mv2_vbuf_ctrl_buf, 64, (sizeof(struct vbuf) *
                    rdma_polling_set_limit * num_rdma_buffer))) {
        PRINT_ERROR("malloc failed: vbuf control struct\n");
        goto fn_fail;
    }
    MPIU_Memset(mv2_vbuf_ctrl_buf, 0, (sizeof(struct vbuf) *
                rdma_polling_set_limit * num_rdma_buffer));

    /* allocate vbuf RDMA buffers */
    if(MPIU_Memalign((void **)&mv2_vbuf_rdma_buf, pagesize,
                (rdma_fp_buffer_size * rdma_polling_set_limit * num_rdma_buffer))) {
        PRINT_ERROR("malloc failed: vbuf DMA buffers\n");
        goto fn_fail;
    }
#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && rdma_eager_cudahost_reg) {
        ibv_cuda_register(mv2_vbuf_rdma_buf, (rdma_fp_buffer_size *
                            rdma_polling_set_limit * num_rdma_buffer));
    }
#endif
    MPIU_Memset(mv2_vbuf_rdma_buf, 0, (rdma_fp_buffer_size *
                rdma_polling_set_limit * num_rdma_buffer));

    mv2_rdma_fp_dreg = dreg_register(mv2_vbuf_rdma_buf, (rdma_fp_buffer_size *
                                    rdma_polling_set_limit * num_rdma_buffer));
    if (mv2_rdma_fp_dreg == NULL) {
        PRINT_ERROR("Dreg register for vbuf DMA buffers\n");
        goto fn_fail;
    }

fn_exit:
    return mpi_errno;

fn_fail:
    if (mv2_vbuf_ctrl_buf) {
        MPIU_Memalign_Free(mv2_vbuf_ctrl_buf);
    }
    if (mv2_vbuf_rdma_buf) {
        MPIU_Memalign_Free(mv2_vbuf_rdma_buf);
    }
    mpi_errno = MPI_ERR_OTHER;
    goto fn_exit;
}

int mv2_free_prealloc_rdma_fp_bufs()
{
    if (mv2_vbuf_ctrl_buf) {
        MPIU_Memalign_Free(mv2_vbuf_ctrl_buf);
        mv2_vbuf_ctrl_buf = NULL;
    }
    if (mv2_vbuf_rdma_buf) {
        dreg_unregister(mv2_rdma_fp_dreg);
        mv2_rdma_fp_dreg = NULL;
        MPIU_Memalign_Free(mv2_vbuf_rdma_buf);
        mv2_vbuf_rdma_buf = NULL;
    }

    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME vbuf_fast_rdma_alloc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int vbuf_fast_rdma_alloc (MPIDI_VC_t * c, int dir)
{
    vbuf * v;
    int i;
    struct ibv_mr *mem_handle[MAX_NUM_HCAS];

    void *vbuf_ctrl_buf = NULL;
    void *vbuf_rdma_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_GEN2_VBUF_FAST_RDMA_ALLOC);
    MPIDI_FUNC_ENTER(MPID_GEN2_VBUF_FAST_RDMA_ALLOC);

    /* initialize revelant fields */
    c->mrail.rfp.rdma_credit = 0;

    if (num_rdma_buffer) {

    if (mv2_rdma_fast_path_preallocate_buffers &&
        (mv2_rdma_fp_prealloc_buf_index < rdma_polling_set_limit)) {
        int ctrl_buf_offset = (sizeof(struct vbuf) * num_rdma_buffer) * mv2_rdma_fp_prealloc_buf_index;
        int data_buf_offset = (num_rdma_buffer * rdma_fp_buffer_size) * mv2_rdma_fp_prealloc_buf_index;

        vbuf_ctrl_buf = (void*) mv2_vbuf_ctrl_buf + ctrl_buf_offset;
        vbuf_rdma_buf = (void*) mv2_vbuf_rdma_buf + data_buf_offset;

        for (i = 0 ; i < rdma_num_hcas; i++) {
            mem_handle[i] = mv2_rdma_fp_dreg->memhandle[i];
        }
        mv2_rdma_fp_prealloc_buf_index++;
    } else {
	/* allocate vbuf struct buffers */
        if(MPIU_Memalign((void **) &vbuf_ctrl_buf, 64,
            sizeof(struct vbuf) * num_rdma_buffer)) {
            DEBUG_PRINT("malloc failed: vbuf in vbuf_fast_rdma_alloc\n");
            goto fn_fail;
        }

        MPIU_Memset(vbuf_ctrl_buf, 0,
                sizeof(struct vbuf) * num_rdma_buffer);

        int pagesize = getpagesize();
        /* allocate vbuf RDMA buffers */
        if(MPIU_Memalign((void **)&vbuf_rdma_buf, pagesize,
                            rdma_fp_buffer_size * num_rdma_buffer)) {
            DEBUG_PRINT("malloc failed: vbuf DMA in vbuf_fast_rdma_alloc");
            goto fn_exit;
        }
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda && rdma_eager_cudahost_reg) {
            ibv_cuda_register(vbuf_rdma_buf, num_rdma_buffer * rdma_fp_buffer_size);
        }
#endif

        MPIU_Memset(vbuf_rdma_buf, 0, rdma_fp_buffer_size * num_rdma_buffer);

        /* REGISTER RDMA SEND BUFFERS */
        for ( i = 0 ; i < rdma_num_hcas; i ++ ) {
            mem_handle[i] =  register_memory(vbuf_rdma_buf,
                                rdma_fp_buffer_size * num_rdma_buffer, i);
            if (!mem_handle[i]) {
                DEBUG_PRINT("fail to register rdma memory, size %d\n",
                            rdma_fp_buffer_size * num_rdma_buffer);
                goto fn_fail;
            }
        }
    }

        /* Connect the DMA buffer to the vbufs */
        for (i = 0; i < num_rdma_buffer; i++) {
            v = ((vbuf *)vbuf_ctrl_buf) + i;
            v->head_flag = (VBUF_FLAG_TYPE *) ( (char *)(vbuf_rdma_buf) + (i *
                          rdma_fp_buffer_size ) );
            v->buffer = (unsigned char *) ( (char *)(vbuf_rdma_buf) + (i *
                          rdma_fp_buffer_size ) + sizeof(*v->head_flag) );
            v->vc     = c;
            v->transport = IB_TRANSPORT_RC;
        }

        /* Some vbuf initialization */
        for (i = 0; i < num_rdma_buffer; i++) {
            if (dir==0) {
                ((vbuf *)vbuf_ctrl_buf + i)->desc.next = NULL;
                ((vbuf *)vbuf_ctrl_buf + i)->padding = FREE_FLAG;
            } else {
                ((vbuf *)vbuf_ctrl_buf + i)->padding = BUSY_FLAG;
            }
        }

        DEBUG_PRINT("[remote-rank %d][dir=%d]"
                "rdma buffer %p, lkey %08x, rkey %08x\n",
                c->pg_rank, dir, vbuf_rdma_buf, mem_handle[0]->lkey,
		mem_handle[0]->rkey);
        if (dir==0) {
            c->mrail.rfp.RDMA_send_buf       = vbuf_ctrl_buf;
            c->mrail.rfp.RDMA_send_buf_DMA   = vbuf_rdma_buf;
            for (i = 0; i < rdma_num_hcas; i++)
                c->mrail.rfp.RDMA_send_buf_mr[i] = mem_handle[i];
            /* set pointers */
            c->mrail.rfp.phead_RDMA_send = 0;
            c->mrail.rfp.ptail_RDMA_send = num_rdma_buffer - 1;
        } else {
            c->mrail.rfp.RDMA_recv_buf       = vbuf_ctrl_buf;
            c->mrail.rfp.RDMA_recv_buf_DMA   = vbuf_rdma_buf;
            for (i = 0; i < rdma_num_hcas; i++)
                c->mrail.rfp.RDMA_recv_buf_mr[i] = mem_handle[i];
        }

    }
fn_exit:
    MPIDI_FUNC_EXIT(MPID_GEN2_VBUF_FAST_RDMA_ALLOC);
    return mpi_errno;
fn_fail:
    if (!mv2_rdma_fast_path_preallocate_buffers) {
        if (vbuf_rdma_buf) {
            MPIU_Memalign_Free(vbuf_rdma_buf);
        }
        if (vbuf_ctrl_buf) {
            MPIU_Memalign_Free(vbuf_ctrl_buf);
        }
    }
    mpi_errno = -1;
    goto fn_exit;
}

