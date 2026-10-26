#!/usr/bin/env python3
"""DGX-852 · Prueba de red del all-reduce RoCE enrutado por par (triángulo TP=3).

Corre DENTRO de la imagen ghcr.io/ursuciprian/spark-vllm-b12x en los tres Sparks a la
vez, con el parche patch_roce_ring.py ya aplicado en cada contenedor. Comprueba la
API real de b12x (RoceOneshotAllReduce + plan preparado) sobre tamaños de 4 KB a 2 MB
contra la suma esperada, mide latencia y muestra el tráfico por HCA (las dos puertas
del rango deben cargar tráfico: una por vecino).

Rangos (ursu-3x.yaml): dgx2=0 (head), dgx1=1, dgx3=2. Bootstrap CPU por la LAN con
gloo, como serve-3x.sh: master spark2 (dgx2), interfaz enP7s7. El transporte
del all-reduce es el anillo ConnectX-7 elegido por B12X_ROCE_HCA + B12X_ROCE_PEER_HCAS
(roce_ring_map.py, mitad A de ring.conf).

Lanzamiento (los tres a la vez; Dani los lanza):

  T=/home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3/roce-anillo
  # con los tres pods de fork-tp3 arriba (aplicar-3x.sh) y ESTE directorio copiado:
  for p in $(kubectl -n llm get pods -l dgx-852=fork-tp3 -o name); do
    kubectl -n llm cp $T ${p#pod/}:/tmp/roce-anillo &
  done; wait
  # en CADA pod (los tres en paralelo; NODE_RANK y QWEN_NCCL_RING_HOST ya vienen del manifiesto):
  B12X_ROCE_RING=1 python3 -I /tmp/roce-anillo/patch_roce_ring.py
  eval "$(python3 -I /tmp/roce-anillo/roce_ring_map.py --host "$QWEN_NCCL_RING_HOST")"
  NODE_RANK=$NODE_RANK MASTER_ADDR=spark2 MASTER_PORT=25001 \
    GLOO_SOCKET_IFNAME=enP7s7 B12X_ROCE_SPIN_LIMIT=30000000 \
    python3 -I /tmp/roce-anillo/test_roce_ring.py

Sin B12X_ROCE_PEER_HCAS la prueba aborta: en el triángulo el all-reduce identidad
cuelga el spin del kernel (mitad del payload sale por la puerta del otro vecino).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

WORLD = 3
HOSTS = ("dgx2", "dgx1", "dgx3")  # rangos 0, 1, 2 (ursu-3x.yaml)
SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
LAT_SIZES = [4096, 65536, 262144, 1048576, 2097152]


def abort(msg: str) -> None:
    print(f"ABORTA: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def detect_gid_index(hcas: list[str]) -> int:
    """Primer GID RoCE v2 IPv4 común a las HCAs (el mismo barrido que serve-3x.sh)."""
    found = None
    for hca in hcas:
        port = Path(f"/sys/class/infiniband/{hca}/ports/1")
        gid = None
        for idx in sorted((port / "gid_attrs" / "types").iterdir(), key=lambda p: int(p.name)):
            if idx.read_text().strip() != "RoCE v2":
                continue
            g = (port / "gids" / idx.name).read_text().strip()
            if g.startswith("0000:0000:0000:0000:0000:ffff:"):
                gid = int(idx.name)
                break
        if gid is None:
            abort(f"{hca}: sin GID RoCE v2 IPv4")
        if found is None:
            found = gid
        elif gid != found:
            abort(f"índice de GID distinto entre HCAs ({hcas}): {found} vs {gid}")
    return found


def main() -> int:
    rank = int(os.environ["NODE_RANK"])
    if not os.environ.get("B12X_ROCE_PEER_HCAS"):
        abort("falta B12X_ROCE_PEER_HCAS: genera las variables del rango con "
              "roce_ring_map.py --host $QWEN_NCCL_RING_HOST")
    # preflight del parche: sin él el proxy de la imagen (ABI 3) empareja por índice
    # y esto cuelga el spin en vez de probar nada
    dist_roce = Path("/usr/local/lib/python3.12/dist-packages/b12x/comm/roce")
    if not (dist_roce / "_roce_proxy.c").is_file():
        dist_roce = Path(sys.prefix) / "lib/python3.12/site-packages/b12x/comm/roce"
    if "B12X_ROCE_RING_POST" not in (dist_roce / "_roce_proxy.c").read_text():
        abort("el parche no está aplicado en este contenedor: "
              "B12X_ROCE_RING=1 python3 -I /tmp/roce-anillo/patch_roce_ring.py")

    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=WORLD)
    print(f"[rank {rank} · {HOSTS[rank]}] gloo arriba con {WORLD} rangos", flush=True)

    from b12x.comm.roce import _preparation as prep
    from b12x.comm.roce._tuning import TUNING
    from b12x.comm.roce.api import AllReduce, discover_hcas

    hcas = list(discover_hcas())
    if len(hcas) != 2:
        abort(f"se esperan 2 HCAs por rango (una por vecino), descubierto: {hcas}")
    if not os.environ.get("B12X_ROCE_GID_INDEX"):
        os.environ["B12X_ROCE_GID_INDEX"] = str(detect_gid_index(hcas))
    print(f"[rank {rank}] hcas={hcas} gid={os.environ['B12X_ROCE_GID_INDEX']} "
          f"map={os.environ['B12X_ROCE_PEER_HCAS']}", flush=True)

    runtime = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device="cuda:0",
        max_size=SIZES[-1],
    )
    query = prep.query_from_runtime(
        runtime,
        surface="AllReduce.all_reduce",
        call={"dtypes": ["float32", "bfloat16"]},
        topology="roce_rdma",
        peer_hosts=HOSTS,
    )
    programs = prep.compile_roce(TUNING.encode_query(query), 0)
    gather_launcher = programs.pop("gather")
    runtime._prepare_resources((torch.float32, torch.bfloat16))
    prepared = prep._PreparedRoce(runtime, programs, gather_launcher)
    dist.barrier()

    # -- corrección: cada rango aporta (rank+1) + i*0.001; la suma esperada es fija --
    bad = 0
    for nbytes in SIZES:
        numel = nbytes // 4
        idx = torch.arange(numel, dtype=torch.float32, device="cuda:0")
        inp = idx * 0.001 + float(rank + 1)
        out = prepared.all_reduce(inp)
        torch.cuda.synchronize()
        expected = idx * 0.003 + 6.0
        if not torch.allclose(out, expected, atol=1e-3, rtol=1e-5):
            bad += 1
            worst = (out - expected).abs().max().item()
            print(f"[rank {rank}] FALLO {nbytes} B: desvío máximo {worst:g}", flush=True)
        # y en bfloat16, el dtype de serving: patrón constante (i*0.001 no es
        # representable en bf16 a 6000 y el cotejo sería falso; 3 y 6 sí)
        numel16 = nbytes // 2
        inp16 = torch.full((numel16,), float(rank + 1), dtype=torch.bfloat16,
                           device="cuda:0")
        out16 = prepared.all_reduce(inp16)
        torch.cuda.synchronize()
        if not bool((out16 == 6.0).all().item()):
            bad += 1
            print(f"[rank {rank}] FALLO bf16 {nbytes} B: "
                  f"{out16.float().min().item():g}..{out16.float().max().item():g}", flush=True)
    verdict = [None] * WORLD
    dist.all_gather_object(verdict, bad)
    if rank == 0:
        print(f"CORRECCION: {'OK' if sum(verdict) == 0 else f'FALLOS {verdict}'} "
              f"(4 KB..2 MB, fp32 y bf16)", flush=True)

    # -- latencia --
    lat = {}
    for nbytes in LAT_SIZES:
        inp = torch.full((nbytes // 2,), float(rank + 1), dtype=torch.bfloat16,
                         device="cuda:0")
        for _ in range(10):
            prepared.all_reduce(inp)
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(100):
            prepared.all_reduce(inp)
        stop.record()
        torch.cuda.synchronize()
        lat[nbytes] = start.elapsed_time(stop) * 1000.0 / 100.0  # µs/op
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, lat)
    if rank == 0:
        print(f"{'tamaño':>10}  " + "  ".join(f"{h:>8}" for h in HOSTS) + "   µs/op", flush=True)
        for nbytes in LAT_SIZES:
            row = "  ".join(f"{g[nbytes]:8.1f}" for g in gathered)
            print(f"{nbytes:>10}  {row}", flush=True)

    stats = [None] * WORLD
    dist.all_gather_object(stats, {
        "rank": rank,
        "hcas": runtime.stats()["hcas"],
        "bytes_posted_per_hca": runtime.stats()["bytes_posted_per_hca"],
        "writes_completed_per_hca": runtime.stats()["writes_completed_per_hca"],
    })
    if rank == 0:
        for s in stats:
            print(f"STATS rank {s['rank']}: {s['bytes_posted_per_hca']} B por HCA "
                  f"{s['hcas']} (terminaciones {s['writes_completed_per_hca']})", flush=True)
        print(f"  las dos puertas del rango deben traer bytes > 0: una por vecino", flush=True)

    runtime.check_health()
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    if rank == 0:
        print(f"ROCE_RING_TEST ok en los 3 rangos ({time.strftime('%H:%M:%S')})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
