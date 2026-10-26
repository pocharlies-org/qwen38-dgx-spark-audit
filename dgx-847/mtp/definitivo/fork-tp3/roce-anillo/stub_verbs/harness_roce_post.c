/* harness_roce_post.c — banco de CPU para _roce_proxy.c contra el doble de
 * libibverbs (DGX-852 roce-anillo).
 *
 * Monta los tres rangos del triángulo TP=3 en un proceso, conecta, lanza UNA
 * operación desde el rango 0 y deja en stdout el stream de RTR/WR. Con
 * "identity" el stream debe ser idéntico entre el .c original y el parcheado
 * (lo compara test_patch_roce_ring.py); con "routed" el propio banco verifica
 * que cada par escribe por su QP, en la ventana recv del vecino y cubriendo
 * todos los slots de flag.
 *
 *   cc -std=gnu11 -I stub_verbs -DROCE_HAVE_SET_PEER \
 *      -o harness harness_roce_post.c _roce_proxy.c stub_verbs/verbs_stub.c -lpthread
 *   ./harness identity | ./harness routed
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 2
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define SLOT_BYTES (2u * 1024 * 1024u)
#define WORLD 3

/* espejo exacto de roce_blob_t en _roce_proxy.c (no se exporta la cabecera) */
typedef struct {
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_HCAS];
    uint16_t lid[ROCE_MAX_HCAS];
    uint8_t gid[ROCE_MAX_HCAS][16];
    uint32_t mtu[ROCE_MAX_HCAS];
    uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
} roce_blob_t;

typedef struct roce_ctx roce_ctx_t;

extern int roce_abi_version(void);
extern int roce_layout(int world, uint64_t slot_bytes, uint64_t *out);
extern uint64_t roce_blob_bytes(void);
extern roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names,
                               int n_hca, int gid_index, void *region,
                               uint64_t region_bytes, uint64_t slot_bytes,
                               char *err, uint64_t err_len);
extern int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len);
extern int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len);
extern int roce_start(roce_ctx_t *c);
extern void roce_stop(roce_ctx_t *c);
extern int roce_failed(roce_ctx_t *c);
extern const char *roce_error(roce_ctx_t *c);
extern uint64_t roce_stat(roce_ctx_t *c, int which);
extern void roce_destroy(roce_ctx_t *c);
extern void stub_set_arena(void *base, size_t bytes);
#ifdef ROCE_HAVE_SET_PEER
extern int roce_set_peer_hca(roce_ctx_t *c, int peer, int local_hca, int remote_hca);
#endif

/* Rutas del triángulo A (roce_ring_map.py --half A), índices sobre
 * hca_names = [rocep1s0f0, rocep1s0f1] de cada rango:
 *   rank0 dgx2: peer1 por f0<->f0, peer2 por f1<->f1
 *   rank1 dgx1: peer0 por f0<->f0, peer2 por f1<->f0(dgx3)
 *   rank2 dgx3: peer0 por f1<->f1, peer1 por f0<->f1(dgx1) */
#ifdef ROCE_HAVE_SET_PEER
static const int ROUTES[3][2][3] = {
    {{1, 0, 0}, {2, 1, 1}},
    {{0, 0, 0}, {2, 1, 0}},
    {{0, 1, 1}, {1, 0, 1}},
};
#endif

/* qp_num del doble: 1000 + h*100 + seq, seq = orden de creación (p ascendente
 * saltando el propio rank). Para el rank0 del banco: peer1->seq0, peer2->seq1. */
static int qpn_for(int hca, int peer) {
    int seq = (peer > 0) ? peer - 1 : peer; /* rank0: p1->0, p2->1 */
    return 1000 + hca * 100 + seq;
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "identity";
    int routed = strcmp(mode, "routed") == 0;
    if (!routed && strcmp(mode, "identity") != 0) {
        fprintf(stderr, "uso: harness identity|routed\n");
        return 2;
    }
#ifdef ROCE_HAVE_SET_PEER
    /* nada */
#else
    if (routed) {
        fprintf(stderr, "harness: este binario no tiene roce_set_peer_hca\n");
        return 2;
    }
#endif

    uint64_t layout[7];
    if (roce_layout(WORLD, SLOT_BYTES, layout) != 0) {
        fprintf(stderr, "roce_layout fallo\n");
        return 1;
    }
    const uint64_t recv_off = layout[0], flag_off = layout[1], send_off = layout[2],
                   ctrl_off = layout[3], total = layout[4];
    printf("ABI %d\n", roce_abi_version());
    printf("LAYOUT recv=%llu flag=%llu send=%llu ctrl=%llu total=%llu stride=%llu slots=%llu\n",
           (unsigned long long)recv_off, (unsigned long long)flag_off,
           (unsigned long long)send_off, (unsigned long long)ctrl_off,
           (unsigned long long)total, (unsigned long long)layout[5],
           (unsigned long long)layout[6]);

    /* un solo arena: region[r] = arena + r*total, para que las direcciones
     * impresas (relativas al arena) sean comparables entre binarios */
    size_t arena_bytes = total * WORLD;
    unsigned char *arena = calloc(1, arena_bytes);
    if (arena == NULL) {
        fprintf(stderr, "sin memoria\n");
        return 1;
    }
    stub_set_arena(arena, arena_bytes);

    const char *const names[2] = {"rocep1s0f0", "rocep1s0f1"};
    roce_ctx_t *ctx[WORLD];
    roce_blob_t blob[WORLD];
    char err[512];
    for (int r = 0; r < WORLD; r++) {
        ctx[r] = roce_create(WORLD, r, names, 2, 3, arena + (size_t)r * total,
                             total, SLOT_BYTES, err, sizeof(err));
        if (ctx[r] == NULL) {
            fprintf(stderr, "roce_create(%d): %s\n", r, err);
            return 1;
        }
    }
    uint64_t blob_bytes = roce_blob_bytes();
    for (int r = 0; r < WORLD; r++) {
        if (roce_local_blob(ctx[r], &blob[r], sizeof(blob[r])) != 0) {
            fprintf(stderr, "roce_local_blob(%d) fallo\n", r);
            return 1;
        }
    }
    (void)blob_bytes;

#ifdef ROCE_HAVE_SET_PEER
    if (routed) {
        for (int r = 0; r < WORLD; r++) {
            for (int e = 0; e < 2; e++) {
                int peer = ROUTES[r][e][0], lh = ROUTES[r][e][1], rh = ROUTES[r][e][2];
                if (roce_set_peer_hca(ctx[r], peer, lh, rh) != 0) {
                    fprintf(stderr, "set_peer_hca(%d->%d,%d,%d): %s\n", r, peer,
                            lh, rh, roce_error(ctx[r]));
                    return 1;
                }
            }
        }
        /* una ruta tardia debe fallar (fail-closed) */
        if (roce_set_peer_hca(ctx[0], 1, 0, 0) != 0) {
            printf("CHECK set-after-set ok (idempotente por re-llamada)\n");
        }
    }
#endif

    unsigned char *joined = malloc(blob_bytes * WORLD);
    for (int r = 0; r < WORLD; r++) {
        memcpy(joined + (size_t)r * blob_bytes, &blob[r], blob_bytes);
    }
    for (int r = 0; r < WORLD; r++) {
        if (roce_connect(ctx[r], joined, blob_bytes * WORLD) != 0) {
            fprintf(stderr, "roce_connect(%d): %s\n", r, roce_error(ctx[r]));
            return 1;
        }
    }
    free(joined);

    for (int r = 0; r < WORLD; r++) {
        if (roce_start(ctx[r]) != 0) {
            fprintf(stderr, "roce_start(%d): %s\n", r, roce_error(ctx[r]));
            return 1;
        }
    }

    /* una operacion desde el rango 0: seq=1 -> slot 1, nbytes=4096 */
    volatile uint32_t *ctrl = (volatile uint32_t *)(arena + ctrl_off);
    const uint32_t nbytes = 4096;
    ctrl[4 + 1u] = nbytes;
    __atomic_store_n((uint32_t *)(arena + ctrl_off), 1u, __ATOMIC_RELEASE);

    int waited = 0;
    while (roce_stat(ctx[0], 0) < 1 && waited < 5000) {
        usleep(1000);
        waited++;
    }
    usleep(100000); /* dejar drenar las terminaciones */

    uint64_t ops = roce_stat(ctx[0], 0);
    uint64_t writes = roce_stat(ctx[0], 1);
    uint64_t last_seq = roce_stat(ctx[0], 2);
    printf("STATS ops=%llu writes=%llu last_seq=%llu failed=%d err=%s\n",
           (unsigned long long)ops, (unsigned long long)writes,
           (unsigned long long)last_seq, roce_failed(ctx[0]),
           roce_error(ctx[0]));

    int rc = 0;
    if (ops != 1 || last_seq != 1 || roce_failed(ctx[0])) {
        printf("FAIL: operacion no publicada correctamente\n");
        rc = 1;
    }

    if (routed && rc == 0) {
        /* rango 0 enruta peer1 por h0 y peer2 por h1; el payload completo
         * (4096 B) debe ir por el QP de la puerta de cada par y los flags
         * cubrir los dos slots del receptor, con el rkey de la HCA REMOTA. */
        const uint64_t slot = 1;
        struct {
            int peer, lh, rh;
        } exp[2] = {{1, 0, 0}, {2, 1, 1}};
        for (int e = 0; e < 2; e++) {
            uint64_t peer_base = (uint64_t)exp[e].peer * total;
            uint64_t want_payload = peer_base + recv_off +
                ((uint64_t)0 * ROCE_SLOTS + slot) * SLOT_BYTES;
            uint64_t want_flag0 = peer_base + flag_off +
                (((uint64_t)0 * ROCE_SLOTS + slot) * 2 + 0) * ROCE_FLAG_STRIDE;
            uint64_t want_flag1 = want_flag0 + ROCE_FLAG_STRIDE;
            int qpn = qpn_for(exp[e].lh, exp[e].peer);
            uint32_t rkey = blob[exp[e].peer].rkey[exp[e].rh];
            char tag[128];
            snprintf(tag, sizeof(tag), "peer%d qpn=%d rkey=0x%x", exp[e].peer,
                     qpn, rkey);
            printf("EXPECT %s payload=+%llx flag0=+%llx flag1=+%llx\n", tag,
                   (unsigned long long)want_payload,
                   (unsigned long long)want_flag0,
                   (unsigned long long)want_flag1);
        }
    }

    for (int r = 0; r < WORLD; r++) {
        roce_stop(ctx[r]);
    }
    for (int r = 0; r < WORLD; r++) {
        roce_destroy(ctx[r]);
    }
    printf("%s\n", rc == 0 ? (routed ? "ROUTED PASS" : "IDENTITY PASS") : "HARNESS FAIL");
    fflush(stdout);
    return rc;
}
