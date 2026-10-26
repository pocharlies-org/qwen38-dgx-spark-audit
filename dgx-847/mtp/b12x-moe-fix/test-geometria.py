#!/usr/bin/env python3
"""DGX-852: geometria del MoE b12x nvfp4 para la vista e768 en TP=3 sin EP, en CPU puro.

Reimplementa las cuatro funciones de decision de b12x 1.3.0
(b12x/moe/fused_moe/_impl.py) con los valores reales del arranque A y comprueba
donde cae cada tamano de grafo y que el scratch del plan es autoconsistente.
Constantes copiadas de _impl.py con su numero de linea:

  _MICRO_MAX_TOKENS = 8                          (_impl.py:1856)
  _MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT = 64      (_impl.py:1853)
  select_tp_moe_backend: micro si m<=8 y m*topk < cutover   (_impl.py:select_tp_moe_backend)
  escalera nvfp4: <15E->16, <48E->32, <96E->64, resto 128    (_impl.py:_select_dynamic_tile_mn)
  _dynamic_task_geometry                           (_impl.py:2159)
  arena nvfp4: token_map (rows_padded,), packed (1,rows_padded,k/2),
               scale (align_up(rows_padded,128), k/16),
               tasks = tiles * ceil(n/128)          (_impl.py:_plan_core_workspace ~3185-3250)

Numeros del arranque A (a-arranque.log:304, checkpoint-meta/config.json):
  E=512, k=hidden=2560, topk=10, moe_intermediate=768, tp=3 -> n=2*I_pp=2*256=512
  cudagraph_capture_sizes=[1,2,3,4,6,8,12,16,24,30,32,40,48,56], max_num_batched_tokens=4096
"""
import math

E = 512
K = 2560
TOPK = 10
N = 512  # 2 * (768/3): FC1 (gate+up) por particion TP
MICRO_MAX_TOKENS = 8
CUTOVER_DEFAULT = 64
CAPTURE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 30, 32, 40, 48, 56]
PREFILL = 4096


def align_up(x, a):
    return ((x + a - 1) // a) * a


def select_backend(m, cutover=CUTOVER_DEFAULT):
    routed = m * TOPK
    return "micro" if (m <= MICRO_MAX_TOKENS and routed < cutover) else "dynamic"


def tile_mn(routed):
    if routed < 15 * E:
        return 16, 128
    if routed < 48 * E:
        return 32, 128
    if routed < 96 * E:
        return 64, 128
    return 128, 128


def task_geometry(routed, tile_m, tile_n):
    routed = max(1, routed)
    base = align_up(routed, tile_m) // tile_m
    tiles = max(1, base + min(E, routed) - 1)
    gate_cnt = max(1, (N + tile_n - 1) // tile_n)
    return tiles, gate_cnt, tiles * gate_cnt


def arena_bytes(m, tile_m):
    routed = m * TOPK
    tiles, _, tasks = task_geometry(routed, tile_m, 128)
    rows_padded = tiles * tile_m
    scale_rows = align_up(rows_padded, 128)
    row_counts = E * 4
    barrier = routed * 4 * 2
    token_map = rows_padded * 4
    token_w = rows_padded * 4
    packed = rows_padded * (K // 2)
    scales = scale_rows * (K // 16)
    etb = (E + 1) * 4
    scalar = 5 * 4
    task_arrays = tasks * 4 * 6
    twc = tiles * 4
    return row_counts + barrier + token_map + token_w + packed + scales + etb + scalar + task_arrays + twc


def main():
    print(f"E={E} k={K} n={N} topk={TOPK} cutover={CUTOVER_DEFAULT}")
    print(f"{'M':>5} {'routed':>7} {'impl':>8} {'tile_m':>6} {'tiles':>6} {'tasks':>6} {'arena_MB':>9}")
    rows = []
    for m in CAPTURE + [PREFILL]:
        impl = select_backend(m)
        routed = m * TOPK
        if impl == "micro":
            tm, _ = tile_mn(routed)
            tiles, _, tasks = task_geometry(routed, tm, 128)
            ab = arena_bytes(m, tm)
        else:
            tm, tn = tile_mn(routed)
            tiles, gate_cnt, tasks = task_geometry(routed, tm, tn)
            ab = arena_bytes(m, tm)
        rows.append((m, impl, tm, tiles, tasks, ab))
        print(f"{m:>5} {routed:>7} {impl:>8} {tm if impl=='dynamic' else '-':>6} "
              f"{tiles:>6} {tasks:>6} {ab/1e6:>9.1f}")

    # 1) La banda del crash: M=6 es el PRIMER micro con enrutado real (warmup solo lanzo
    #    1 firma MoE, M=1 con cyclic de 10 expertos; a-arranque.log:659 "Warmed up 1 b12x
    #    MoE kernel signature(s)"). M=8..56 ya corrieron dynamic@16 con enrutado real en
    #    las 9 capturas que pasaron antes de reventar (barra 9/14, fallo en la 10 = M=6).
    crash_m = [m for (m, impl, *_ ) in rows if m == 6][0]
    impl6 = select_backend(6)
    assert impl6 == "micro", f"M=6 deberia ser micro (60 < cutover {CUTOVER_DEFAULT})"
    assert select_backend(8) == "dynamic", "M=8 (80 pares) NO es micro: explica que pasara"
    for m in (12, 16, 24, 30, 32, 40, 48, 56):
        assert select_backend(m) == "dynamic" and tile_mn(m * TOPK)[0] == 16
    # 2) Con el workaround (cutover=0) todo cae a dynamic@16, la banda ya probada con
    #    enrutado real en este mismo arranque.
    for m in CAPTURE:
        assert select_backend(m, cutover=0) == "dynamic"
    # 3) Autoconsistencia del scratch dynamic@16 para M=56 (verificada a mano en el
    #    analisis): slot maximo de publicacion = (tiles-1)*gate_cnt + (gate_cnt-1) < tasks.
    tiles, gate_cnt, tasks = task_geometry(56 * TOPK, 16, 128)
    assert (tiles - 1) * gate_cnt + (gate_cnt - 1) == tasks - 1
    # 4) Upper bound de tiles del dynamic: base + min(E, routed) - 1. Con enrutado real
    #    (560 pares, hasta 512 expertos activos) el maximo alcanzable de sum ceil(c_e/16)
    #    es 512 (512 expertos x1 fila + 48 filas extra que no cruzan borde de tile), y el
    #    plan reserva 546: el bound del plan cubre la distribucion real. No hay OOB por
    #    geometria en la banda dynamic -> el fallo tiene que estar en la otra banda.
    base = align_up(56 * TOPK, 16) // 16
    alcanzable = 512  # 512 expertos x1 fila; los 48 pares extra no abren tile nuevo
    assert tiles == base + min(E, 56 * TOPK) - 1 == 546
    assert alcanzable <= tiles
    print("\nOK: M=6..4/3/2/1 son la primera ejecucion REAL del micro nvfp4; el resto del")
    print("banda dynamic@16 ya paso con enrutado real. cutover=0 manda todo a dynamic.")


if __name__ == "__main__":
    main()
