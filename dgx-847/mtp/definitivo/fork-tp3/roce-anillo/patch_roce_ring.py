#!/usr/bin/env python3
"""DGX-852 · B12X_ROCE_RING: all-reduce RoCE del fork de ursuciprian en el TRIÁNGULO TP=3.

Se ejecuta al arrancar el contenedor, ANTES de `vllm serve`, sobre los ficheros
instalados en /usr/local/lib/python3.12/dist-packages (igual que patch_gdn_mix.py).
No toca el checkpoint ni el cluster. El proxy C se recompila solo en el arranque:
b12x/comm/roce/_proxy.py lo builda por sha256 del fuente, así que editar el .c
produce una .so nueva sin tocar nada más.

Qué cambia y por qué. El all-reduce one-shot empareja cada par de rangos por
ÍNDICE de HCA (`_roce_proxy.c`: `memcpy(blob.gid[h], c->hca[h].gid.raw,16)` /
`peer->gid[h]`, `qp_num[h][p]`, `peer_rkey[h][p]`): asume un switch donde el
hca-i de cada nodo llega al hca-i del vecino. En el triángulo sin switch cada
enlace es un puerto concreto (ring.conf, INFRA-741): dgx1<->dgx2 por el puerto 0
(10.0.0.x/10.0.1.x), dgx1<->dgx3 por 10.0.2.x/10.0.3.x (rocep1s0f1 de dgx1 contra
rocep1s0f0 de dgx3) y dgx2<->dgx3 por 10.0.4.x/10.0.5.x. Con ROCE_MAX_HCAS=2 cada
rango abre exactamente dos HCAs, una por vecino, y el índice local NO coincide
con el remoto para todos los pares: eso es lo que cuelga el spin del kernel.

El parche añade ENRUTADO POR PAR:
  - C: `roce_set_peer_hca(ctx, peer, local_hca, remote_hca)` fija para ese par
    UNA pareja (HCA local, HCA remoto); el connect y el post usan la pareja.
    El flag lo escribe en TODOS los slots 0..n_hca-1 del receptor (su kernel
    espera todos), en el mismo QP y detrás del payload: la entrega ordenada de
    RC garantiza payload-antes-que-flag. Solo el último flag va señalizado
    (RC completa en orden: una terminación cubre el payload y los previos).
  - Python: `B12X_ROCE_PEER_HCAS="<peer>:<hca-local>:<hca-remoto>[,...]"` por
    rango (lo genera roce_ring_map.py desde ring.conf). Cada rango publica sus
    nombres de HCA y su mapa en un intercambio glosado ANTES del connect y la
    ruta se rechaza si falta un par, si un nombre no existe en el lado que dice,
    o si la simetría no casa (mi remoto = su local y a la inversa).
  - ABI: ROCE_ABI_VERSION 3 -> 4 (cambia la semántica de connect; el check de
    `_proxy.py` y el `proxy_abi` del config colectivo viajan con ella).

SIN B12X_ROCE_PEER_HCAS el comportamiento es IDÉNTICO al de la imagen: no se
declara ninguna ruta, `peer_n_hca[p] = n_hca` con identidad h->h, y el bucle de
post recorre exactamente los mismos WR (mismo QP, misma dirección, mismo rkey,
mismo flag) que el original. La prueba de CPU lo comprueba comparando el stream
de `ibv_post_send` contra un doble de libibverbs, original vs parcheado.

Idempotente (marca por edición), anclado con count(old)==1, FAIL-CLOSED: si un
ancla no casa exactamente una vez sale con 3 y no escribe NINGÚN fichero (las
tres ediciones van en memoria; se escriben todas o ninguna). Con B12X_ROCE_RING
distinto de 1 sale con 2; vacío: inerte, exit 0.

Uso:
  B12X_ROCE_RING=1 python3 -I patch_roce_ring.py                      # en la imagen
  B12X_ROCE_RING=1 python3 -I patch_roce_ring.py --root <dist-packages> --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARK = "B12X_ROCE_RING"

ROCE_C = "b12x/comm/roce/_roce_proxy.c"
PROXY_PY = "b12x/comm/roce/_proxy.py"
ONESHOT_PY = "b12x/comm/roce/roce_oneshot.py"

# ---------------------------------------------------------------------------
# Anclas. Cada edición: (ruta, viejo, nuevo, marca). `nuevo` lleva `marca`
# para que la idempotencia se detecte buscándola.
# ---------------------------------------------------------------------------

EDITS: list[tuple[str, str, str, str]] = [
    # (C1) ABI 3 -> 4: la semántica de roce_connect cambia (empareja por ruta).
    (
        ROCE_C,
        "#define ROCE_ABI_VERSION 3\n",
        "#define ROCE_ABI_VERSION 4  // B12X_ROCE_RING_ABI: enrutado de HCA por par\n",
        "B12X_ROCE_RING_ABI",
    ),
    # (C2) Campo `connected` para congelar las rutas tras el connect.
    (
        ROCE_C,
        "    int started;\n",
        "    int started;\n"
        "    int connected;  // B12X_ROCE_RING_CONNFIELD: rutas congeladas tras connect\n",
        "B12X_ROCE_RING_CONNFIELD",
    ),
    # (C3) Tabla de enrutado por par en el contexto.
    (
        ROCE_C,
        "    uint64_t peer_addr[ROCE_MAX_PEERS];\n"
        "    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS];\n",
        "    uint64_t peer_addr[ROCE_MAX_PEERS];\n"
        "    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS];\n"
        "    // B12X_ROCE_RING_ROUTE: enrutado por par.  Por defecto identidad\n"
        "    // (stripe del payload por las n_hca HCAs, h local -> h remoto), que es\n"
        "    // la malla con switch de la imagen original.  roce_set_peer_hca fija\n"
        "    // para un par UNA pareja (HCA local, HCA remoto): el triángulo sin\n"
        "    // switch, donde cada enlace es un puerto concreto de ring.conf.\n"
        "    uint8_t peer_n_hca[ROCE_MAX_PEERS];\n"
        "    uint8_t peer_h_local[ROCE_MAX_PEERS][ROCE_MAX_HCAS];\n"
        "    uint8_t peer_h_remote[ROCE_MAX_PEERS][ROCE_MAX_HCAS];\n",
        "B12X_ROCE_RING_ROUTE",
    ),
    # (C4) roce_create: enrutado identidad por defecto (TP=2 idéntico a la imagen).
    (
        ROCE_C,
        "    for (int h = 0; h < n_hca; h++) {\n"
        "        if (open_hca(c, h, hca_names[h]) != 0) {\n"
        "            snprintf(err, err_len, \"%s\", c->err);\n"
        "            roce_destroy(c);\n"
        "            return NULL;\n"
        "        }\n"
        "    }\n"
        "    return c;\n"
        "}\n",
        "    for (int h = 0; h < n_hca; h++) {\n"
        "        if (open_hca(c, h, hca_names[h]) != 0) {\n"
        "            snprintf(err, err_len, \"%s\", c->err);\n"
        "            roce_destroy(c);\n"
        "            return NULL;\n"
        "        }\n"
        "    }\n"
        "    // B12X_ROCE_RING_INIT: identidad salvo que roce_set_peer_hca diga otra\n"
        "    // cosa; sin B12X_ROCE_PEER_HCAS el camino es el de la imagen original.\n"
        "    for (int p = 0; p < world; p++) {\n"
        "        c->peer_n_hca[p] = (uint8_t)n_hca;\n"
        "        for (int h = 0; h < n_hca; h++) {\n"
        "            c->peer_h_local[p][h] = (uint8_t)h;\n"
        "            c->peer_h_remote[p][h] = (uint8_t)h;\n"
        "        }\n"
        "    }\n"
        "    return c;\n"
        "}\n",
        "B12X_ROCE_RING_INIT",
    ),
    # (C5) roce_set_peer_hca, exportada; antes de connect y antes de start.
    (
        ROCE_C,
        "int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {\n",
        "// B12X_ROCE_RING_SETFN: fija el par (peer -> una HCA local + una remota).\n"
        "// Triángulo sin switch: cada vecino se ve por un puerto concreto, así que\n"
        "// el stripe por índice de HCA de la malla no vale. Debe llamarse tras\n"
        "// roce_create y antes de roce_connect; sin llamarlo queda la identidad.\n"
        "int roce_set_peer_hca(roce_ctx_t *c, int peer, int local_hca, int remote_hca) {\n"
        "    if (c->started || c->connected || peer < 0 || peer >= c->world ||\n"
        "        peer == c->rank || local_hca < 0 || local_hca >= c->n_hca ||\n"
        "        remote_hca < 0 || remote_hca >= c->n_hca) {\n"
        "        snprintf(c->err, sizeof(c->err),\n"
        "                 \"invalid RoCE peer route (peer %d local %d remote %d, \"\n"
        "                 \"started %d connected %d)\", peer, local_hca, remote_hca,\n"
        "                 c->started, c->connected);\n"
        "        return -1;\n"
        "    }\n"
        "    c->peer_n_hca[peer] = 1;\n"
        "    c->peer_h_local[peer][0] = (uint8_t)local_hca;\n"
        "    c->peer_h_remote[peer][0] = (uint8_t)remote_hca;\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {\n",
        "B12X_ROCE_RING_SETFN",
    ),
    # (C6) connect_qp: el otro extremo se lee por el índice REMOTO de la ruta.
    (
        ROCE_C,
        "static int connect_qp(roce_ctx_t *c, int h, int p, const roce_blob_t *peer) {\n"
        "    roce_hca_t *hca = &c->hca[h];\n"
        "    struct ibv_qp_attr rtr;\n"
        "    memset(&rtr, 0, sizeof(rtr));\n"
        "    rtr.qp_state = IBV_QPS_RTR;\n"
        "    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[h] < (uint32_t)hca->mtu ? peer->mtu[h] : (uint32_t)hca->mtu);\n"
        "    rtr.dest_qp_num = peer->qp_num[h][c->rank];\n"
        "    rtr.rq_psn = 0;\n"
        "    rtr.max_dest_rd_atomic = 1;\n"
        "    rtr.min_rnr_timer = 12;\n"
        "    rtr.ah_attr.is_global = 1;\n"
        "    rtr.ah_attr.dlid = peer->lid[h];\n"
        "    rtr.ah_attr.sl = 0;\n"
        "    rtr.ah_attr.src_path_bits = 0;\n"
        "    rtr.ah_attr.port_num = ROCE_PORT;\n"
        "    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[h], 16);\n",
        "static int connect_qp(roce_ctx_t *c, int h, int p, const roce_blob_t *peer,\n"
        "                      int rh) {  // B12X_ROCE_RING_CONNECT: h local, rh remoto\n"
        "    roce_hca_t *hca = &c->hca[h];\n"
        "    struct ibv_qp_attr rtr;\n"
        "    memset(&rtr, 0, sizeof(rtr));\n"
        "    rtr.qp_state = IBV_QPS_RTR;\n"
        "    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[rh] < (uint32_t)hca->mtu ? peer->mtu[rh] : (uint32_t)hca->mtu);\n"
        "    rtr.dest_qp_num = peer->qp_num[rh][c->rank];\n"
        "    rtr.rq_psn = 0;\n"
        "    rtr.max_dest_rd_atomic = 1;\n"
        "    rtr.min_rnr_timer = 12;\n"
        "    rtr.ah_attr.is_global = 1;\n"
        "    rtr.ah_attr.dlid = peer->lid[rh];\n"
        "    rtr.ah_attr.sl = 0;\n"
        "    rtr.ah_attr.src_path_bits = 0;\n"
        "    rtr.ah_attr.port_num = ROCE_PORT;\n"
        "    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[rh], 16);\n",
        "B12X_ROCE_RING_CONNECT",
    ),
    # (C7) roce_connect: conecta solo las parejas enrutadas y congela las rutas.
    (
        ROCE_C,
        "        c->peer_addr[p] = all[p].region_addr;\n"
        "        for (int h = 0; h < c->n_hca; h++) {\n"
        "            c->peer_rkey[h][p] = all[p].rkey[h];\n"
        "            if (connect_qp(c, h, p, &all[p]) != 0) {\n"
        "                return -1;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    return 0;\n"
        "}\n",
        "        c->peer_addr[p] = all[p].region_addr;\n"
        "        for (int h = 0; h < c->n_hca; h++) {\n"
        "            c->peer_rkey[h][p] = all[p].rkey[h];\n"
        "        }\n"
        "        // B12X_ROCE_RING_CONNECTLOOP: con enrutado identidad (sin la variable\n"
        "        // de entorno) conecta exactamente los mismos QP que el original.\n"
        "        for (int k = 0; k < c->peer_n_hca[p]; k++) {\n"
        "            if (connect_qp(c, c->peer_h_local[p][k], p, &all[p],\n"
        "                           c->peer_h_remote[p][k]) != 0) {\n"
        "                return -1;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    c->connected = 1;\n"
        "    return 0;\n"
        "}\n",
        "B12X_ROCE_RING_CONNECTLOOP",
    ),
    # (C8) post_op: stripe por las HCAs ENRUTADAS del par; el flag cubre todos
    # los slots del receptor cuando el par va por una sola HCA.
    (
        ROCE_C,
        "        uint32_t pack_offset = 0;\n"
        "        for (int h = 0; h < c->n_hca; h++) {\n"
        "            roce_hca_t *hca = &c->hca[h];\n"
        "            // Two work requests per stripe; keep the queue at most a quarter\n"
        "            // full so a provider that needs extra entries can never fail a post.\n"
        "            while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {\n"
        "                if (drain_cq(c, h) != 0) {\n"
        "                    return -1;\n"
        "                }\n"
        "                // A peer that stopped acknowledging keeps the QP retrying for\n"
        "                // a long time; honour a stop request instead of blocking\n"
        "                // roce_stop (and so teardown) behind it.\n"
        "                if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {\n"
        "                    snprintf(c->err, sizeof(c->err),\n"
        "                             \"RoCE proxy stopped with %u writes outstanding to rank %d\",\n"
        "                             hca->outstanding[p], p);\n"
        "                    return -1;\n"
        "                }\n"
        "            }\n"
        "            uint32_t stripe_packs = total_packs / (uint32_t)c->n_hca;\n"
        "            if ((uint32_t)h < total_packs % (uint32_t)c->n_hca) {\n"
        "                stripe_packs += 1;\n"
        "            }\n"
        "            uint32_t stripe_bytes = stripe_packs * 16;\n"
        "            uint64_t byte_offset = (uint64_t)pack_offset * 16;\n"
        "            uint64_t remote = c->peer_addr[p];\n"
        "            struct ibv_sge flag_sge = {\n"
        "                .addr = (uint64_t)(uintptr_t)&seq_copy,\n"
        "                .length = 4,\n"
        "                .lkey = 0,\n"
        "            };\n"
        "            struct ibv_send_wr flag_wr;\n"
        "            memset(&flag_wr, 0, sizeof(flag_wr));\n"
        "            flag_wr.wr_id = (uint64_t)p;\n"
        "            flag_wr.sg_list = &flag_sge;\n"
        "            flag_wr.num_sge = 1;\n"
        "            flag_wr.opcode = IBV_WR_RDMA_WRITE;\n"
        "            flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;\n"
        "            flag_wr.wr.rdma.remote_addr =\n"
        "                remote + c->flag_off +\n"
        "                (((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_hca +\n"
        "                 (uint64_t)h) * ROCE_FLAG_STRIDE;\n"
        "            flag_wr.wr.rdma.rkey = c->peer_rkey[h][p];\n"
        "\n"
        "            struct ibv_send_wr data_wr;\n"
        "            struct ibv_sge data_sge;\n"
        "            struct ibv_send_wr *first_wr = &flag_wr;\n"
        "            if (stripe_bytes != 0) {\n"
        "                data_sge = (struct ibv_sge){\n"
        "                    .addr = (uint64_t)(uintptr_t)(send + byte_offset),\n"
        "                    .length = stripe_bytes,\n"
        "                    .lkey = hca->mr->lkey,\n"
        "                };\n"
        "                memset(&data_wr, 0, sizeof(data_wr));\n"
        "                data_wr.wr_id = (uint64_t)p;\n"
        "                data_wr.next = &flag_wr;\n"
        "                data_wr.sg_list = &data_sge;\n"
        "                data_wr.num_sge = 1;\n"
        "                data_wr.opcode = IBV_WR_RDMA_WRITE;\n"
        "                data_wr.wr.rdma.remote_addr =\n"
        "                    remote + c->recv_off +\n"
        "                    ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes +\n"
        "                    byte_offset;\n"
        "                data_wr.wr.rdma.rkey = c->peer_rkey[h][p];\n"
        "                first_wr = &data_wr;\n"
        "            }\n"
        "            struct ibv_send_wr *bad = NULL;\n"
        "            int rc = ibv_post_send(hca->qp[p], first_wr, &bad);\n"
        "            if (rc != 0) {\n"
        "                set_err(c, \"ibv_post_send\", rc);\n"
        "                return -1;\n"
        "            }\n"
        "            hca->outstanding[p] += 1;\n"
        "            hca->bytes_posted += stripe_bytes;\n"
        "            pack_offset += stripe_packs;\n"
        "        }\n",
        "        uint32_t pack_offset = 0;\n"
        "        // B12X_ROCE_RING_POST: el stripe recorre las HCAs ENRUTADAS del par.\n"
        "        // Con enrutado identidad (sin B12X_ROCE_PEER_HCAS) route_n == n_hca,\n"
        "        // h == rh == k y cada WR es idéntico al del bucle original: mismo QP,\n"
        "        // misma dirección remota, mismo rkey, mismo flag señalizado. Con el\n"
        "        // triángulo enrutado route_n == 1: el payload entero va por la única\n"
        "        // puerta del par y el flag se escribe en TODOS los slots 0..n_hca-1\n"
        "        // del receptor (su kernel espera todos), en el mismo QP y detrás del\n"
        "        // payload; RC entrega en orden, así que el payload precede a cada\n"
        "        // flag. Solo el último va señalizado: una terminación en orden cubre\n"
        "        // el payload y los flags previos, y el control de flujo sigue\n"
        "        // contando una unidad por stripe como antes.\n"
        "        int route_n = c->peer_n_hca[p];\n"
        "        for (int k = 0; k < route_n; k++) {\n"
        "            int h = c->peer_h_local[p][k];\n"
        "            int rh = c->peer_h_remote[p][k];\n"
        "            roce_hca_t *hca = &c->hca[h];\n"
        "            // Two work requests per stripe; keep the queue at most a quarter\n"
        "            // full so a provider that needs extra entries can never fail a post.\n"
        "            while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {\n"
        "                if (drain_cq(c, h) != 0) {\n"
        "                    return -1;\n"
        "                }\n"
        "                // A peer that stopped acknowledging keeps the QP retrying for\n"
        "                // a long time; honour a stop request instead of blocking\n"
        "                // roce_stop (and so teardown) behind it.\n"
        "                if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {\n"
        "                    snprintf(c->err, sizeof(c->err),\n"
        "                             \"RoCE proxy stopped with %u writes outstanding to rank %d\",\n"
        "                             hca->outstanding[p], p);\n"
        "                    return -1;\n"
        "                }\n"
        "            }\n"
        "            uint32_t stripe_packs = total_packs / (uint32_t)route_n;\n"
        "            if ((uint32_t)k < total_packs % (uint32_t)route_n) {\n"
        "                stripe_packs += 1;\n"
        "            }\n"
        "            uint32_t stripe_bytes = stripe_packs * 16;\n"
        "            uint64_t byte_offset = (uint64_t)pack_offset * 16;\n"
        "            uint64_t remote = c->peer_addr[p];\n"
        "            int n_flag = (route_n == c->n_hca) ? 1 : c->n_hca;\n"
        "            struct ibv_sge flag_sge = {\n"
        "                .addr = (uint64_t)(uintptr_t)&seq_copy,\n"
        "                .length = 4,\n"
        "                .lkey = 0,\n"
        "            };\n"
        "            struct ibv_send_wr flag_wr[ROCE_MAX_HCAS];\n"
        "            for (int f = 0; f < n_flag; f++) {\n"
        "                memset(&flag_wr[f], 0, sizeof(flag_wr[f]));\n"
        "                flag_wr[f].wr_id = (uint64_t)p;\n"
        "                flag_wr[f].sg_list = &flag_sge;\n"
        "                flag_wr[f].num_sge = 1;\n"
        "                flag_wr[f].opcode = IBV_WR_RDMA_WRITE;\n"
        "                flag_wr[f].send_flags = IBV_SEND_INLINE |\n"
        "                    ((f + 1 == n_flag) ? IBV_SEND_SIGNALED : 0);\n"
        "                flag_wr[f].wr.rdma.remote_addr =\n"
        "                    remote + c->flag_off +\n"
        "                    (((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_hca +\n"
        "                     (uint64_t)((route_n == c->n_hca) ? h : f)) * ROCE_FLAG_STRIDE;\n"
        "                flag_wr[f].wr.rdma.rkey = c->peer_rkey[rh][p];\n"
        "                flag_wr[f].next = (f + 1 < n_flag) ? &flag_wr[f + 1] : NULL;\n"
        "            }\n"
        "\n"
        "            struct ibv_send_wr data_wr;\n"
        "            struct ibv_sge data_sge;\n"
        "            struct ibv_send_wr *first_wr = &flag_wr[0];\n"
        "            if (stripe_bytes != 0) {\n"
        "                data_sge = (struct ibv_sge){\n"
        "                    .addr = (uint64_t)(uintptr_t)(send + byte_offset),\n"
        "                    .length = stripe_bytes,\n"
        "                    .lkey = hca->mr->lkey,\n"
        "                };\n"
        "                memset(&data_wr, 0, sizeof(data_wr));\n"
        "                data_wr.wr_id = (uint64_t)p;\n"
        "                data_wr.next = &flag_wr[0];\n"
        "                data_wr.sg_list = &data_sge;\n"
        "                data_wr.num_sge = 1;\n"
        "                data_wr.opcode = IBV_WR_RDMA_WRITE;\n"
        "                data_wr.wr.rdma.remote_addr =\n"
        "                    remote + c->recv_off +\n"
        "                    ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes +\n"
        "                    byte_offset;\n"
        "                data_wr.wr.rdma.rkey = c->peer_rkey[rh][p];\n"
        "                first_wr = &data_wr;\n"
        "            }\n"
        "            struct ibv_send_wr *bad = NULL;\n"
        "            int rc = ibv_post_send(hca->qp[p], first_wr, &bad);\n"
        "            if (rc != 0) {\n"
        "                set_err(c, \"ibv_post_send\", rc);\n"
        "                return -1;\n"
        "            }\n"
        "            hca->outstanding[p] += 1;\n"
        "            hca->bytes_posted += stripe_bytes;\n"
        "            pack_offset += stripe_packs;\n"
        "        }\n",
        "B12X_ROCE_RING_POST",
    ),
    # (P1) _proxy.py: el check de ABI viaja con el .c.
    (
        PROXY_PY,
        "        if lib.roce_abi_version() != 3:\n",
        "        if lib.roce_abi_version() != 4:  # B12X_ROCE_RING_ABI_PY: enrutado por par\n",
        "B12X_ROCE_RING_ABI_PY",
    ),
    # (P2) _proxy.py: binding de la nueva llamada.
    (
        PROXY_PY,
        "        lib.roce_local_blob.restype = ctypes.c_int\n",
        "        lib.roce_set_peer_hca.restype = ctypes.c_int\n"
        "        lib.roce_set_peer_hca.argtypes = [\n"
        "            p,\n"
        "            ctypes.c_int,\n"
        "            ctypes.c_int,\n"
        "            ctypes.c_int,\n"
        "        ]  # B12X_ROCE_RING_BIND\n"
        "        lib.roce_local_blob.restype = ctypes.c_int\n",
        "B12X_ROCE_RING_BIND",
    ),
    # (P3) _proxy.py: método de enrutado, antes de connect.
    (
        PROXY_PY,
        "    def connect(self, blobs: list[bytes]) -> None:\n",
        "    def set_peer_hcas(self, routes: dict[int, tuple[int, int]]) -> None:\n"
        "        \"\"\"B12X_ROCE_RING_ROUTES: fija por par una HCA local y una remota.\n"
        "\n"
        "        Debe llamarse antes de ``connect``; sin llamarlo el proxy mantiene\n"
        "        el enrutado identidad de la malla con switch (TP=2 sin cambios).\n"
        "        \"\"\"\n"
        "        for peer, (local_hca, remote_hca) in sorted(routes.items()):\n"
        "            if self._lib.roce_set_peer_hca(\n"
        "                self._ctx, int(peer), int(local_hca), int(remote_hca)\n"
        "            ) != 0:\n"
        "                raise RuntimeError(f\"RoCE peer route failed: {self.error()}\")\n"
        "\n"
        "    def connect(self, blobs: list[bytes]) -> None:\n",
        "B12X_ROCE_RING_ROUTES",
    ),
    # (O1) roce_oneshot.py: parser y validación cruzada del mapa por par.
    (
        ONESHOT_PY,
        "def is_supported(device: torch.device | int | str | None = None) -> bool:\n",
        "# B12X_ROCE_RING_FUNCS: mapa por par para el triangulo sin switch (DGX-852).\n"
        "def parse_peer_hcas(raw: str) -> dict[int, tuple[str, str]]:\n"
        "    \"\"\"Lee ``B12X_ROCE_PEER_HCAS``: ``peer:hca-local:hca-remota`` separados por comas.\n"
        "\n"
        "    Los nombres se resuelven contra el ``hca_names`` de cada rango en\n"
        "    ``validate_peer_routes``; vacio significa enrutado identidad (TP=2).\n"
        "    \"\"\"\n"
        "    out: dict[int, tuple[str, str]] = {}\n"
        "    for item in raw.split(\",\"):\n"
        "        item = item.strip()\n"
        "        if not item:\n"
        "            continue\n"
        "        parts = [p.strip() for p in item.split(\":\")]\n"
        "        if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:\n"
        "            raise ValueError(\n"
        "                \"B12X_ROCE_PEER_HCAS: expected peer:local_hca:remote_hca, \"\n"
        "                f\"got {item!r}\"\n"
        "            )\n"
        "        try:\n"
        "            peer = int(parts[0])\n"
        "        except ValueError as exc:\n"
        "            raise ValueError(\n"
        "                f\"B12X_ROCE_PEER_HCAS: rank is not an integer in {item!r}\"\n"
        "            ) from exc\n"
        "        if peer in out:\n"
        "            raise ValueError(f\"B12X_ROCE_PEER_HCAS: duplicate rank {peer}\")\n"
        "        out[peer] = (parts[1], parts[2])\n"
        "    return out\n"
        "\n"
        "\n"
        "def validate_peer_routes(\n"
        "    rank: int,\n"
        "    world_size: int,\n"
        "    hca_names: Sequence[str],\n"
        "    env_map: dict[int, tuple[str, str]],\n"
        "    announcements: list[dict],\n"
        ") -> dict[int, tuple[int, int]]:\n"
        "    \"\"\"B12X_ROCE_RING_VALIDATE: coteja el mapa contra lo que publica cada rango.\n"
        "\n"
        "    ``announcements[peer]`` es ``{\"hca_names\": [...], \"peer_hcas\": {peer:\n"
        "    [local, remota]}}`` del propio peer.  Falla (RuntimeError) si falta un\n"
        "    par, sobra uno, un nombre no existe en el lado que dice o la simetría\n"
        "    no casa (mi remota = su local y a la inversa).  Devuelve\n"
        "    ``{peer: (indice_local, indice_remoto)`` para el proxy.\n"
        "    \"\"\"\n"
        "    peers = {p for p in range(world_size) if p != rank}\n"
        "    if set(env_map) != peers:\n"
        "        raise RuntimeError(\n"
        "            f\"B12X_ROCE_PEER_HCAS on rank {rank} covers ranks \"\n"
        "            f\"{sorted(env_map)}, expected {sorted(peers)}\"\n"
        "        )\n"
        "    routes: dict[int, tuple[int, int]] = {}\n"
        "    for peer in sorted(peers):\n"
        "        local_name, remote_name = env_map[peer]\n"
        "        if local_name not in hca_names:\n"
        "            raise RuntimeError(\n"
        "                f\"B12X_ROCE_PEER_HCAS on rank {rank}: local HCA \"\n"
        "                f\"{local_name!r} for rank {peer} is not one of {list(hca_names)}\"\n"
        "            )\n"
        "        peer_names = list(announcements[peer][\"hca_names\"])\n"
        "        if remote_name not in peer_names:\n"
        "            raise RuntimeError(\n"
        "                f\"B12X_ROCE_PEER_HCAS on rank {rank}: remote HCA \"\n"
        "                f\"{remote_name!r} for rank {peer} is not one of rank {peer}'s \"\n"
        "                f\"{peer_names}\"\n"
        "            )\n"
        "        theirs = announcements[peer][\"peer_hcas\"].get(rank)\n"
        "        if theirs is None:\n"
        "            raise RuntimeError(\n"
        "                f\"B12X_ROCE_PEER_HCAS: rank {peer} has no route back to rank {rank}\"\n"
        "            )\n"
        "        if [theirs[0], theirs[1]] != [remote_name, local_name]:\n"
        "            raise RuntimeError(\n"
        "                f\"B12X_ROCE_PEER_HCAS asymmetric on link {rank}<->{peer}: \"\n"
        "                f\"rank {rank} routes {local_name!r}->{remote_name!r}, rank \"\n"
        "                f\"{peer} routes {theirs[0]!r}->{theirs[1]!r}\"\n"
        "            )\n"
        "        routes[peer] = (\n"
        "            list(hca_names).index(local_name),\n"
        "            peer_names.index(remote_name),\n"
        "        )\n"
        "    return routes\n"
        "\n"
        "\n"
        "def is_supported(device: torch.device | int | str | None = None) -> bool:\n",
        "B12X_ROCE_RING_FUNCS",
    ),
    # (O2) roce_oneshot.py: leer el mapa (dentro del try: un env roto se reporta
    # por el canal colectivo, como un fallo del proxy; ningún rango se queda solo).
    (
        ONESHOT_PY,
        "        error: Optional[str] = None\n"
        "        try:\n"
        "            self._proxy = Proxy(\n",
        "        error: Optional[str] = None\n"
        "        peer_raw = os.getenv(\"B12X_ROCE_PEER_HCAS\", \"\").strip()\n"
        "        self._peer_env: dict[int, tuple[str, str]] = {}\n"
        "        try:\n"
        "            if peer_raw:\n"
        "                # B12X_ROCE_RING_PARSE: malformado -> error colectivo, no hang.\n"
        "                self._peer_env = parse_peer_hcas(peer_raw)\n"
        "            self._proxy = Proxy(\n",
        "B12X_ROCE_RING_PARSE",
    ),
    # (O3) roce_oneshot.py: el config colectivo exige que el mapa esté presente
    # en todos los rangos o en ninguno (una mezcla colgaría el intercambio extra).
    (
        ONESHOT_PY,
        "            \"hca_count\": len(self.hca_names),\n",
        "            \"hca_count\": len(self.hca_names),\n"
        "            \"peer_hcas_present\": bool(peer_raw),\n"
        "            # B12X_ROCE_RING_CFG: el mapa por par es todo o nada.\n",
        "B12X_ROCE_RING_CFG",
    ),
    # (O4) roce_oneshot.py: intercambio de anuncios + validación + rutas, antes
    # del connect. Con env vacío este bloque no se ejecuta: TP=2 byte a byte igual.
    (
        ONESHOT_PY,
        "        try:\n"
        "            self._proxy.connect([s[1] for s in statuses])\n"
        "            self._proxy.start()\n"
        "        except Exception as exc:  # noqa: BLE001\n"
        "            error = str(exc)\n",
        "        try:\n"
        "            if self._peer_env:\n"
        "                # B12X_ROCE_RING_ROUTES_PY: cada rango publica sus nombres de\n"
        "                # HCA y su mapa; ninguna ruta se conecta si la simetria no\n"
        "                # casa en los dos extremos del cable (fail-closed).\n"
        "                announcements = _exchange(\n"
        "                    {\n"
        "                        \"hca_names\": list(self.hca_names),\n"
        "                        \"peer_hcas\": {\n"
        "                            p: [lh, rh] for p, (lh, rh) in self._peer_env.items()\n"
        "                        },\n"
        "                    },\n"
        "                    self._group,\n"
        "                )\n"
        "                routes = validate_peer_routes(\n"
        "                    self.rank,\n"
        "                    self.world_size,\n"
        "                    self.hca_names,\n"
        "                    self._peer_env,\n"
        "                    announcements,\n"
        "                )\n"
        "                self._proxy.set_peer_hcas(routes)\n"
        "            self._proxy.connect([s[1] for s in statuses])\n"
        "            self._proxy.start()\n"
        "        except Exception as exc:  # noqa: BLE001\n"
        "            error = str(exc)\n",
        "B12X_ROCE_RING_ROUTES_PY",
    ),
]


def resolve_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    default = Path("/usr/local/lib/python3.12/dist-packages")
    if (default / "b12x" / "comm" / "roce" / "_roce_proxy.c").is_file():
        return default
    sys.exit(
        f"{MARK} ABORTA: no encuentro b12x/comm/roce; pasa --root <dist-packages>"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="dist-packages con b12x/. Por defecto el instalado en la imagen.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Comprueba anclas sin escribir.")
    args = ap.parse_args(argv)

    gate = os.environ.get(MARK, "").strip()
    if gate not in ("", "1"):
        print(f"{MARK} ABORTA: valor invalido {gate!r} (esperado 1 o vacio)",
              file=sys.stderr)
        return 2
    if gate == "":
        print(f"{MARK} inerte (vacio): no toco nada")
        return 0

    root = resolve_root(args.root)
    if not (root / "b12x" / "comm" / "roce").is_dir():
        print(f"{MARK} ABORTA: {root} no tiene b12x/comm/roce", file=sys.stderr)
        return 2

    # FASE 1: leer, anclar y compilar TODO en memoria. FASE 2: escribir solo si
    # todo casa (el .c y los dos .py viajan juntos: ABI 4 exige los tres).
    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for rel, old, new, marker in EDITS:
        by_file.setdefault(rel, []).append((old, new, marker))

    applied: list[str] = []
    skipped: list[str] = []
    staged: dict[Path, str] = {}
    for rel, edits in by_file.items():
        path = root / rel
        if not path.is_file():
            print(f"{MARK} ABORTA: falta {path}", file=sys.stderr)
            return 3
        text = path.read_text()
        original = text
        for old, new, marker in edits:
            if marker in text:
                skipped.append(f"{rel}#{marker}")
                continue
            n = text.count(old)
            if n != 1:
                print(f"{MARK} ABORTA: ancla no unica o ausente en {rel} "
                      f"(count={n} para {marker}): {old[:72]!r}", file=sys.stderr)
                return 3
            text = text.replace(old, new)
        if rel.endswith(".py"):
            # Validar sintaxis ANTES de escribir (fail-closed).
            try:
                compile(text, str(path), "exec")
            except SyntaxError as exc:
                print(f"{MARK} ABORTA: el fichero parcheado no compila: "
                      f"{rel}: {exc}", file=sys.stderr)
                return 3
        if text != original:
            staged[path] = text
            for _o, _n, marker in edits:
                if marker not in original:
                    applied.append(f"{rel}#{marker}")

    if not args.dry_run:
        for path, text in staged.items():
            path.write_text(text)

    tag = "aplicado" if applied else "ya aplicado"
    print(f"{MARK}=1 {tag}: " + (", ".join(applied) if applied else "(nada nuevo)"))
    if skipped:
        print(f"{MARK}: ya estaban: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
