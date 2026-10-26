/* verbs_stub.c — implementación del doble de libibverbs (DGX-852 roce-anillo).
 *
 * Dos dispositivos falsos con los nombres reales del Spark (rocep1s0f0,
 * rocep1s0f1), todo ACTIVE, y un registro en stdout de cada WR que pasa por
 * ibv_post_send para que el banco compare streams. Las claves y los números
 * de QP son deterministas:
 *   qp_num  = 1000 + dev_index*100 + seq_qp_en_el_ctx
 *   lkey/rkey = 0x1000 + ctx_index*4 + dev_index   (ctx_index = orden de
 *     ibv_open_device global: en el banco, rank*2 + dev)
 *   gid.raw[15] = ctx_index*2 + dev_index
 */
#include "infiniband/verbs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct ibv_device devices[2] = {
    {"rocep1s0f0"},
    {"rocep1s0f1"},
};
static struct ibv_device *device_ptrs[2] = {&devices[0], &devices[1]};
static int open_count;
static unsigned char *arena_base;
static size_t arena_bytes;

void stub_set_arena(void *base, size_t bytes) {
    arena_base = (unsigned char *)base;
    arena_bytes = bytes;
}

static void print_rel_addr(const char *tag, uint64_t addr) {
    if (arena_base != NULL && addr >= (uint64_t)(uintptr_t)arena_base &&
        addr < (uint64_t)(uintptr_t)arena_base + arena_bytes) {
        printf(" %s=+%llx", tag, (unsigned long long)(addr - (uint64_t)(uintptr_t)arena_base));
    } else {
        printf(" %s=ext%llx", tag, (unsigned long long)addr);
    }
}

struct ibv_device **ibv_get_device_list(int *num_devices) {
    if (num_devices) {
        *num_devices = 2;
    }
    return device_ptrs;
}

void ibv_free_device_list(struct ibv_device **list) {
    (void)list;
}

const char *ibv_get_device_name(struct ibv_device *device) {
    return device->name;
}

struct ibv_context *ibv_open_device(struct ibv_device *device) {
    struct ibv_context *ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->device = device;
    ctx->dev_index = (int)(device - devices);
    ctx->ctx_index = open_count / 2; /* el banco abre (h0,h1) por rank */
    open_count++;
    return ctx;
}

int ibv_close_device(struct ibv_context *context) {
    free(context);
    return 0;
}

int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                   struct ibv_port_attr *port_attr) {
    (void)context;
    (void)port_num;
    memset(port_attr, 0, sizeof(*port_attr));
    port_attr->state = IBV_PORT_ACTIVE;
    port_attr->lid = 0;
    port_attr->active_mtu = IBV_MTU_4096;
    return 0;
}

int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                  union ibv_gid *gid) {
    (void)port_num;
    memset(gid, 0, sizeof(*gid));
    gid->raw[0] = 0xfe;
    gid->raw[15] = (uint8_t)(context->ctx_index * 2 + context->dev_index + index);
    return 0;
}

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
    struct ibv_pd *pd = calloc(1, sizeof(*pd));
    if (pd) {
        pd->ctx = context;
    }
    return pd;
}

int ibv_dealloc_pd(struct ibv_pd *pd) {
    free(pd);
    return 0;
}

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                          int access) {
    (void)access;
    struct ibv_mr *mr = calloc(1, sizeof(*mr));
    if (mr == NULL) {
        return NULL;
    }
    mr->addr = addr;
    mr->length = length;
    mr->lkey = mr->rkey =
        (uint32_t)(0x1000 + pd->ctx->ctx_index * 4 + pd->ctx->dev_index);
    return mr;
}

int ibv_dereg_mr(struct ibv_mr *mr) {
    free(mr);
    return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                             void *cq_context, void *channel, int comp_vector) {
    (void)context;
    (void)cqe;
    (void)cq_context;
    (void)channel;
    (void)comp_vector;
    return calloc(1, sizeof(struct ibv_cq));
}

int ibv_destroy_cq(struct ibv_cq *cq) {
    free(cq);
    return 0;
}

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                             struct ibv_qp_init_attr *qp_init_attr) {
    struct ibv_qp *qp = calloc(1, sizeof(*qp));
    if (qp == NULL) {
        return NULL;
    }
    qp->cq = qp_init_attr->send_cq;
    qp->hca = pd->ctx->dev_index;
    qp->peer = -1;
    qp->qp_num = (uint32_t)(1000 + pd->ctx->dev_index * 100 + pd->ctx->qp_seq);
    pd->ctx->qp_seq++;
    return qp;
}

int ibv_destroy_qp(struct ibv_qp *qp) {
    free(qp);
    return 0;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
    (void)attr_mask;
    if (attr->qp_state == IBV_QPS_RTR) {
        /* El banco imprime el emparejamiento: qué QP local se conecta a qué
         * QP remoto, por qué GID y con qué MTU. Así el diff identidad y las
         * aserciones enrutadas ven también el connect, no solo el post. */
        printf("RTR qpn=%u dest=%u dgid15=%u sgid=%u mtu=%d\n",
               qp->qp_num, attr->dest_qp_num, attr->ah_attr.grh.dgid.raw[15],
               attr->ah_attr.grh.sgid_index, (int)attr->path_mtu);
        fflush(stdout);
    }
    return 0;
}

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr) {
    *bad_wr = NULL;
    for (struct ibv_send_wr *w = wr; w != NULL; w = w->next) {
        uint32_t len = 0;
        uint64_t saddr = 0;
        for (int i = 0; i < w->num_sge; i++) {
            len += w->sg_list[i].length;
            saddr = w->sg_list[i].addr;
        }
        printf("WR qpn=%u len=%u", qp->qp_num, len);
        if (len == 4) {
            printf(" saddr=stack"); /* el seq vive en la pila del proxy */
        } else {
            print_rel_addr("saddr", saddr);
        }
        print_rel_addr("raddr", w->wr.rdma.remote_addr);
        printf(" rkey=0x%x flags=%d wr_id=%llu\n", w->wr.rdma.rkey,
               w->send_flags, (unsigned long long)w->wr_id);
        fflush(stdout);
        if (w->send_flags & IBV_SEND_SIGNALED) {
            struct ibv_cq *cq = qp->cq;
            cq->done[cq->tail % 4096] = w->wr_id;
            cq->tail++;
        }
    }
    return 0;
}

int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    int n = 0;
    while (n < num_entries && cq->head != cq->tail) {
        wc[n].wr_id = cq->done[cq->head % 4096];
        cq->head++;
        wc[n].status = IBV_WC_SUCCESS;
        wc[n].vendor_err = 0;
        n++;
    }
    return n;
}

const char *ibv_wc_status_str(enum ibv_wc_status status) {
    return status == IBV_WC_SUCCESS ? "success" : "stub-error";
}
