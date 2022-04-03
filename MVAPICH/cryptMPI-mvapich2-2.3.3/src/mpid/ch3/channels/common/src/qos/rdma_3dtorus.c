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

#include "rdma_3dtorus.h"

/*
 * The service level is possibly dynamic and if so the only way to get a
 * proper value is to ask the SA.
 *
 * A cache is kept of ib queue pairs used to communicate with the SA for a
 * particular device and port.
 *
 * A cache is kept of path record SL values retrieved from that SA.
 *
 * The interaction with the cache assumes that there are no recursive calls
 * to these routines.  This must be solved either by code flow, by using
 * higher level locks, or by adding a locking mechanism to these routines
 * along with some method for avoiding deadlock.
 *
 * This code was adapted from gPXE's get path record, so we can use a UD
 * queue pair to talk to the SA, and not need to chmod /dev/infiniband/umad*
 * for use by normal users.  This code does not support redirection of the
 * SA's queue pair, and has not been tested on anything other than default
 * subnet configurations.
 *
 * The request to the SA is a SubnAdmGet(), not a SubnAdmGetTable(), since
 * we only need a single path, plus this is what gPXE does.
 *
 * This does not ask for reversible paths, since each peer will call this
 * routine on it's own node, and therefore could use one-way paths if they
 * were returned.
 */

struct openib_sa_qp_cache_t *openib_sa_qp_cache = NULL;

int mv2_get_path_rec_sl(struct ibv_context *context_arg, struct ibv_pd *hca_pd,
                        uint32_t port_num, uint16_t lid, uint16_t rem_lid,
                        int network_is_3dtorus, int num_sa_retries)
{
    struct ibv_context *context;
    struct openib_sa_qp_cache_t *cache;
    struct ibv_pd *pd;
    struct ibv_ah_attr aattr;
    struct ibv_qp_init_attr iattr;
    struct ibv_qp_attr mattr;
    struct ibv_port_attr pattr;
    struct ibv_recv_wr *brwr;
    int i;

    for (cache = openib_sa_qp_cache; cache; cache = cache->next) {
  	    if ((strcmp(cache->device_name,
                    ibv_get_device_name (context_arg->device)) == 0) &&
            (cache->port_num == port_num)) {
                    break;
        }
    }

    /* One time setup for each device/port combination */
    if (!cache) {
    	context = context_arg;
    	if (!context) {
            fprintf(stderr, "Failed to open device");
            goto fn_fail;
    	}

    	if ((cache = MPIU_Malloc(sizeof(struct openib_sa_qp_cache_t))) == NULL) {
            fprintf(stderr, "cannot posix_memalign SA cache\n");
            goto fn_fail;
    	}

    	cache->context = context;
    	cache->device_name = (char *) ibv_get_device_name(context->device);
    	cache->port_num = port_num;

    	for (i = 0; i < sizeof(cache->sl_values); i++) {
            /* Special value means not present */
    	    cache->sl_values[i] = 0x7F;
        }

    	cache->next = openib_sa_qp_cache;
    	openib_sa_qp_cache = cache;
    
    	pd = hca_pd;
    	if (!pd) {
            fprintf(stderr, "Failed to alloc pd number ");
            goto fn_fail;
    	}
    
    	cache->mr = ibv_reg_mr(pd, cache->send_recv_buffer,
    			                sizeof(cache->send_recv_buffer),
    			                IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_LOCAL_WRITE);
    	if (!cache->mr) {
            fprintf(stderr, "Cannot create MR\n");
            goto fn_fail;
    	}
    
    	cache->cq = ibv_create_cq(context, 4, NULL, NULL, 0);
    	if (!cache->cq) {
            fprintf(stderr, "Cannot create CQ\n");
            goto fn_fail;
    	}
    
    	memset(&iattr, 0, sizeof(iattr));
    	iattr.send_cq = cache->cq;
    	iattr.recv_cq = cache->cq;
    	iattr.cap.max_send_wr = 2;
    	iattr.cap.max_recv_wr = 2;
    	iattr.cap.max_send_sge = 1;
    	iattr.cap.max_recv_sge = 1;
    	iattr.qp_type = IBV_QPT_UD;

    	cache->qp = ibv_create_qp(pd, &iattr);
    	if (!cache->qp) {
            fprintf(stderr, "Failed to create qp for SL queries\n");
            goto fn_fail;
    	}
    
    	memset(&mattr, 0, sizeof(mattr));
    	mattr.qp_state = IBV_QPS_INIT;
    	mattr.port_num = port_num;
    	mattr.qkey = IB_GLOBAL_QKEY;

    	if (ibv_modify_qp(cache->qp, &mattr,
    					IBV_QP_STATE              |
    					IBV_QP_PKEY_INDEX         |
    					IBV_QP_PORT               |
    					IBV_QP_QKEY)) {
            fprintf(stderr, "Failed to modify QP to INIT");
            goto fn_fail;
    	}
    
    	if (ibv_query_port(context, port_num, &pattr)) {
            fprintf(stderr, "Failed to query port\n");
            goto fn_fail;
    	}

    	memset(&aattr, 0, sizeof(aattr));
    	aattr.dlid = pattr.sm_lid;
    	aattr.sl = pattr.sm_sl;
    	aattr.port_num = port_num;

    	cache->ah = ibv_create_ah(pd, &aattr);
    	if (!cache->ah) {
            fprintf(stderr, "Failed to create AH for SL queries\n");
            goto fn_fail;
    	}
    
    	memset(&mattr, 0, sizeof(mattr));
    	mattr.qp_state = IBV_QPS_RTR;
    	if (ibv_modify_qp(cache->qp, &mattr, IBV_QP_STATE)) {
            fprintf(stderr, "Failed to modify QP to RTR");
            goto fn_fail;
    	}
    
    	mattr.qp_state = IBV_QPS_RTS;
    	if (ibv_modify_qp(cache->qp, &mattr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
            fprintf(stderr, "Failed to modify QP to RTS");
            goto fn_fail;
    	}
    
    	memset(&(cache->rwr), 0, sizeof(cache->rwr));
    	cache->rwr.num_sge = 1;
    	cache->rwr.sg_list = &(cache->rsge);
    	memset(&(cache->rsge), 0, sizeof(cache->rsge));
    	cache->rsge.addr = (uint64_t)(void *)
    			(cache->send_recv_buffer + sizeof(struct ib_mad_sa));
    	cache->rsge.length = sizeof(struct ib_mad_sa) + 40;
    	cache->rsge.lkey = cache->mr->lkey;

    	if (ibv_post_recv(cache->qp, &(cache->rwr), &brwr)) {
            fprintf(stderr, "Failed to post first recv buffer");
            goto fn_fail;
    	}
    }

    context = cache->context;

    /* If the destination lid SL value is not in the cache, go get it */
    if (cache->sl_values[rem_lid] == 0x7F) {
    	struct ib_mad_sa *sag, *sar;
    	struct ibv_send_wr swr, *bswr;
    	struct ibv_sge ssge;
    	struct ibv_wc wc;
    	int got_sl_value, get_sl_retries;
    
    	/* If we were passed a local LID of 0x0000 (reserved), then we have to
    	   look up the correct value */
    	if (lid == 0x0) {
    	    struct ibv_port_attr port_attr;
    	    int ret;
    
    	    if ((ret = ibv_query_port(context, port_num, &port_attr))) {
                fprintf(stderr, "Failed to query port\n");
                goto fn_fail;
    	    }
    
    	    lid = port_attr.lid;
    	}
    
    	/* *sag is first buffer, where we build the SA Get request to send */
    	sag = (void *)(cache->send_recv_buffer);
    	memset(sag, 0, sizeof(*sag));

    	sag->mad_hdr.base_version       = IB_MGMT_BASE_VERSION;
    	sag->mad_hdr.mgmt_class         = IB_MGMT_CLASS_SUBN_ADM;
    	sag->mad_hdr.class_version      = 2;
    	sag->mad_hdr.method             = IB_MGMT_METHOD_GET;
    	sag->mad_hdr.attr_id            = htons (IB_SA_ATTR_PATH_REC);
    	sag->mad_hdr.tid[0]             = IB_SA_TID_GET_PATH_REC_0 +
                                            cache->qp->qp_num;
    	sag->mad_hdr.tid[1]             = IB_SA_TID_GET_PATH_REC_1 + rem_lid;
    	sag->sa_hdr.comp_mask[1]        = htonl(IB_SA_PATH_REC_DLID |
                                                IB_SA_PATH_REC_SLID);
    	sag->sa_data.path_record.dlid   = htons(rem_lid);
    	sag->sa_data.path_record.slid   = htons(lid);
    
    	memset(&swr, 0, sizeof(swr));
    	memset(&ssge, 0, sizeof(ssge));

    	swr.sg_list                 = &ssge;
    	swr.num_sge                 = 1;
    	swr.opcode                  = IBV_WR_SEND;
    	swr.wr.ud.ah                = cache->ah;
    	swr.wr.ud.remote_qpn        = IB_SA_QPN;
    	swr.wr.ud.remote_qkey       = IB_GLOBAL_QKEY;
    	swr.send_flags              = IBV_SEND_SIGNALED |
                                      IBV_SEND_SOLICITED;

    	ssge.addr = (uint64_t)(void *)sag;
    	ssge.length = sizeof(*sag);
    	ssge.lkey = cache->mr->lkey;
    
    	got_sl_value = 0;
    	get_sl_retries = 0;
    
    	/* *sar is the receive buffer */
    	sar = (void *)(cache->send_recv_buffer + sizeof(struct ib_mad_sa) + 40);
    
    	while (!got_sl_value) {
    	    struct timeval get_sl_last_sent, get_sl_last_poll;
    
    	    if (ibv_post_send(cache->qp, &swr, &bswr)) {
                fprintf(stderr, "Failed to post send");
                goto fn_fail;
    	    }
    	    gettimeofday(&get_sl_last_sent, NULL);
    
    	    while (!got_sl_value) {
        		i = ibv_poll_cq(cache->cq, 1, &wc);
        		if (i > 0 &&
                    (wc.status == IBV_WC_SUCCESS) &&
                    (wc.opcode == IBV_WC_RECV) &&
                    (wc.byte_len >= sizeof(*sar)) &&
                    (sar->mad_hdr.tid[0] == sag->mad_hdr.tid[0]) &&
                    (sar->mad_hdr.tid[1] == sag->mad_hdr.tid[1])) {

        		    if ((sar->mad_hdr.status == 0) &&
        		        (sar->sa_data.path_record.slid == htons(lid)) &&
        		        (sar->sa_data.path_record.dlid == htons(rem_lid))) {
        			    /* Everything matches, so we have the desired SL */
        			    cache->sl_values[rem_lid] =
        			            sar->sa_data.path_record.reserved__sl & 0x0F;
                        /* still must repost recieve buf */
                        got_sl_value = 1;
        		    } else {
                		/* Probably bad status, unlikely bad lid match.  We
                		   will ignore response and let it time out so that
                           we do a retry, but after a delay.  We must make
                           a new TID so the SM doesn't see it as the same
                           request.
                         */
                		sag->mad_hdr.tid[1] += 0x10000;
                	}

                	if (ibv_post_recv(cache->qp, &(cache->rwr), &brwr)) {
                        fprintf(stderr, "Failed to receive buffer");
                        goto fn_fail;
                	}
        		}
        		if (i == 0) {
                    /* poll did not find anything */
        		    gettimeofday(&get_sl_last_poll, NULL);
        		    i = get_sl_last_poll.tv_sec - get_sl_last_sent.tv_sec;
        		    i = (i * 1000000) +
        			    get_sl_last_poll.tv_usec - get_sl_last_sent.tv_usec;
        		    if (i > 2000000) {
                        /* two second timeout for reply */
            			get_sl_retries++;
            			if (get_sl_retries >= num_sa_retries) {
                            /* Default 20 attempts */
                            fprintf(stderr, "No response from SA\n");
                            goto fn_fail;
            			}
            			break;	/* retransmit request */
        		    }
        		    usleep(100);  /* otherwise pause before polling again */
        		}
    	    } /* while (!got_sl_value) */
    	} /* while (!got_sl_value) */
    }

    /* Now all we do is send back the value laying around */
    return cache->sl_values[rem_lid];
fn_fail:
    if (network_is_3dtorus) {
        fprintf(stderr, "Error: Failed to query Subnet Manager for correct "
                "Service Level. This will cause deadlock in a 3D Torus network."
                " Program aborting to prevent deadlock in the system\n");
        exit(1);
    } else {
        fprintf(stderr, "Error: Failed to query Subnet Manager for correct "
                "Service Level. Using default Service Level of 0.\n");
    }
    return 0;
}

int mv2_release_3d_torus_resources()
{
    int err = MPI_SUCCESS;
    struct openib_sa_qp_cache_t *cache = NULL;

    for (cache = openib_sa_qp_cache; cache; cache = cache->next) {
        err = ibv_destroy_ah(cache->ah);

        err = ibv_dereg_mr(cache->mr);

        err = ibv_destroy_cq(cache->cq);

        err = ibv_destroy_qp(cache->qp);
    }

    return err;
}
