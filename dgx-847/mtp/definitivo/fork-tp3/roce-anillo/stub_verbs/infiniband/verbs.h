/* stub_verbs/infiniband/verbs.h — DOBLE de libibverbs para pruebas de CPU
 * (DGX-852 roce-anillo). No es la ABI real: define solo lo que usa
 * b12x/comm/roce/_roce_proxy.c, con los mismos nombres de miembros, para
 * poder compilar el proxy y observar su stream de ibv_post_send en la x86
 * sin RDMA. La compilación real ocurre dentro de la imagen con las cabeceras
 * de verdad (b12x/comm/roce/_proxy.py).
 */
#ifndef STUB_INFINIBAND_VERBS_H
#define STUB_INFINIBAND_VERBS_H

#include <stddef.h>
#include <stdint.h>

enum ibv_port_state { IBV_PORT_DOWN = 1, IBV_PORT_ACTIVE = 4 };
enum ibv_mtu { IBV_MTU_256 = 0, IBV_MTU_2048 = 4, IBV_MTU_4096 = 5 };
enum ibv_qp_state {
    IBV_QPS_RESET = 0, IBV_QPS_INIT = 1, IBV_QPS_RTR = 2, IBV_QPS_RTS = 3
};
enum ibv_qp_type { IBV_QPT_RC = 2 };
enum ibv_access_flags {
    IBV_ACCESS_LOCAL_WRITE = 1, IBV_ACCESS_REMOTE_WRITE = 2
};
enum ibv_wr_opcode { IBV_WR_RDMA_WRITE = 0, IBV_WR_SEND = 1 };
enum ibv_send_flags {
    IBV_SEND_FENCE = 1, IBV_SEND_SIGNALED = 2, IBV_SEND_INLINE = 4
};
enum ibv_wc_status { IBV_WC_SUCCESS = 0, IBV_WC_REM_ACCESS_ERR = 10 };
enum ibv_qp_attr_mask {
    IBV_QP_STATE = 1 << 0,
    IBV_QP_PKEY_INDEX = 1 << 1,
    IBV_QP_PORT = 1 << 2,
    IBV_QP_ACCESS_FLAGS = 1 << 3,
    IBV_QP_AV = 1 << 4,
    IBV_QP_PATH_MTU = 1 << 5,
    IBV_QP_DEST_QPN = 1 << 6,
    IBV_QP_RQ_PSN = 1 << 7,
    IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 8,
    IBV_QP_MIN_RNR_TIMER = 1 << 9,
    IBV_QP_TIMEOUT = 1 << 10,
    IBV_QP_RETRY_CNT = 1 << 11,
    IBV_QP_RNR_RETRY = 1 << 12,
    IBV_QP_SQ_PSN = 1 << 13,
    IBV_QP_MAX_QP_RD_ATOMIC = 1 << 14
};

union ibv_gid {
    uint8_t raw[16];
};

struct ibv_device {
    char name[64];
};

struct ibv_context {
    struct ibv_device *device;
    int dev_index;   /* 0/1 según el orden de ibv_get_device_list */
    int ctx_index;   /* orden de apertura global: rank*2 + dev en el banco */
    int qp_seq;      /* contador de QPs creados en este contexto */
};

struct ibv_pd {
    struct ibv_context *ctx;
};

struct ibv_mr {
    uint32_t lkey;
    uint32_t rkey;
    void *addr;
    size_t length;
};

struct ibv_cq {
    /* FIFO de terminaciones pendientes (wr_id de los WR señalizados) */
    uint64_t done[4096];
    int head, tail;
};

struct ibv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

struct ibv_qp_cap {
    int max_send_wr;
    int max_recv_wr;
    int max_send_sge;
    int max_recv_sge;
    int max_inline_data;
};

struct ibv_qp_init_attr {
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    enum ibv_qp_type qp_type;
    struct ibv_qp_cap cap;
};

struct ibv_global_route {
    union ibv_gid dgid;
    uint32_t flow_label;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
};

struct ibv_ah_attr {
    uint8_t is_global;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t port_num;
    struct ibv_global_route grh;
};

struct ibv_qp_attr {
    enum ibv_qp_state qp_state;
    int qkey;
    uint32_t dest_qp_num;
    uint32_t rq_psn;
    int max_dest_rd_atomic;
    int min_rnr_timer;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    uint32_t sq_psn;
    int timeout;
    int retry_cnt;
    int rnr_retry;
    int max_rd_atomic;
    uint8_t port_num;
    uint16_t pkey_index;
    uint32_t qp_access_flags;
};

struct ibv_qp {
    uint32_t qp_num;
    struct ibv_cq *cq;
    int peer;      /* marcado por el banco: para qué par se creó */
    int hca;       /* índice de HCA del dueño */
};

struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    int send_flags;
    union {
        struct {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
    } wr;
};

struct ibv_port_attr {
    enum ibv_port_state state;
    enum ibv_mtu active_mtu;
    uint16_t lid;
};

struct ibv_wc {
    uint64_t wr_id;
    enum ibv_wc_status status;
    int vendor_err;
};

/* El banco de pruebas enlaza su implementación (verbs_stub.c). */
void stub_set_arena(void *base, size_t bytes); /* para imprimir saddr/raddr relativos */
struct ibv_device **ibv_get_device_list(int *num_devices);
void ibv_free_device_list(struct ibv_device **list);
const char *ibv_get_device_name(struct ibv_device *device);
struct ibv_context *ibv_open_device(struct ibv_device *device);
int ibv_close_device(struct ibv_context *context);
int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                   struct ibv_port_attr *port_attr);
int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                  union ibv_gid *gid);
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);
int ibv_dealloc_pd(struct ibv_pd *pd);
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                          int access);
int ibv_dereg_mr(struct ibv_mr *mr);
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                             void *cq_context, void *channel, int comp_vector);
int ibv_destroy_cq(struct ibv_cq *cq);
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                             struct ibv_qp_init_attr *qp_init_attr);
int ibv_destroy_qp(struct ibv_qp *qp);
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
const char *ibv_wc_status_str(enum ibv_wc_status status);

#endif /* STUB_INFINIBAND_VERBS_H */
