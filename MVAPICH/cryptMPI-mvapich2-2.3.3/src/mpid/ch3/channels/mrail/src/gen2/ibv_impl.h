#ifndef IBV_IMPL_H_
#define IBV_IMPL_H_

void adjust_weights(MPIDI_VC_t *vc, double start_time,
    double *finish_time,
    double *init_weight);

void get_wall_time(double *t);

int perform_manual_apm(struct ibv_qp* qp);

#endif /* IBV_IMPL_H_ */
