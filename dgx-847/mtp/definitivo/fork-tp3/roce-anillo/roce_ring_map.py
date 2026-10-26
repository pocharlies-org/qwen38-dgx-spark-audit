#!/usr/bin/env python3
"""B12X_ROCE_* del triangulo sin switch (DGX-852): mapa por par para el all-reduce RoCE del fork.

Mismo mecanismo que nccl_ring_graph.py de DGX-815 (el que usa QWEN_NCCL_RING_HOST en
serve-3x.sh): lee ring.conf (copia congelada junto a este fichero, o la del repo con
--ring-conf) y el orden del anillo por rango — dgx2, dgx1, dgx3 = rangos 0, 1, 2 de
ursu-3x.yaml — e imprime las variables de ESTE rango:

  B12X_ROCE_RING=1            puerta del parche de arranque (patch_roce_ring.py)
  B12X_ROCE_HCA=<a,b>         las dos HCAs locales del rango, una por enlace, en orden PCI
  B12X_ROCE_PEER_HCAS=<mapa>  peer:hca-local:hca-remota por nombre, un cable por par

Cada Spark abre exactamente dos HCAs (ROCE_MAX_HCAS=2 en _roce_proxy.c) y en el anillo
cada una mira a un vecino distinto: dgx1<->dgx2 va por el puerto 0 (10.0.0.x / 10.0.1.x),
dgx1<->dgx3 por 10.0.2.x / 10.0.3.x (rocep1s0f1 de dgx1 contra rocep1s0f0 de dgx3) y
dgx2<->dgx3 por 10.0.4.x / 10.0.5.x. Se usa la mitad A de cada par; --half B cambia
todas las puertas a la otra raiz PCIe (0002:01:00.x). Un enlace por par = 200 Gb/s por
vecino: el stripe de dos mitades a un mismo vecino solo existe en TP=2, donde los dos
puertos 0 unen el mismo par.

El mapa va por NOMBRES, no por indices: el runtime resuelve cada nombre contra el
hca_names de su rango y exige en el intercambio de arranque que la simetria casa
(mi remota = su local y a la inversa), asi que un orden de HCA distinto o un
B12X_ROCE_HCA reordenado no cuela una ruta cruzada.

  python3 -I roce_ring_map.py --host dgx2              # lineas KEY=VALUE de este rango
  python3 -I roce_ring_map.py --all                    # los tres rangos (para el yaml)
  python3 -I roce_ring_map.py --check                  # simetria cruzada de los tres
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ANILLO = ("dgx2", "dgx1", "dgx3")  # rangos 0, 1 y 2 del TP=3 (ursu-3x.yaml)
COLUMNAS = ("kind", "host", "netdev", "rdma", "cidr", "peer", "half", "uuid", "name")
RING_CONF = Path(__file__).with_name("ring.conf")  # copia congelada; --ring-conf para otra


def leer_ring_conf(ruta: Path):
    """Las filas de ring.conf como dicts. Las `tp` traen `-` en `rdma`: es el de la misma funcion PCIe."""
    filas = []
    for n, linea in enumerate(Path(ruta).read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip() or linea.startswith("#"):
            continue
        campos = linea.split()
        if len(campos) != len(COLUMNAS):
            raise ValueError(f"{ruta}:{n}: {len(campos)} columnas y ring.conf tiene {len(COLUMNAS)}")
        fila = dict(zip(COLUMNAS, campos, strict=True))
        if fila["rdma"] == "-":
            fila["rdma"] = re.sub(r"np\d+$", "", re.sub(r"^en", "roce", fila["netdev"]))
        filas.append(fila)
    return filas


def pci(rdma: str):
    """(dominio, bus, slot, funcion) del nombre RDMA: rocep1s0f1 = 0000:01:00.1, roceP2p1s0f1 = 0002:01:00.1."""
    dom, bus, slot, fn = re.fullmatch(r"roce(?:P(\d+))?p(\d+)s(\d+)f(\d+)", rdma).groups()
    return int(dom or 0), int(bus), int(slot), int(fn)


def puerta(filas, host, vecino, half):
    """La funcion RDMA de `host` que mira a `vecino` por la mitad A o B del par (ring.conf)."""
    rdma = {f["half"]: f["rdma"] for f in filas if f["host"] == host and f["peer"] == vecino}
    if half not in rdma:
        raise ValueError(f"ring.conf no da la mitad {half} de {host} hacia {vecino}: hay {sorted(rdma)}")
    return rdma[half]


def mapa_rango(filas, host, half):
    """(rank, hca_names ordenadas por PCI, {peer: (local, remota)}) del rango que corre en `host`."""
    if host not in ANILLO:
        raise ValueError(f"{host} no esta en el anillo {','.join(ANILLO)}")
    rank = ANILLO.index(host)
    entradas = {}
    locales = set()
    for p, ph in enumerate(ANILLO):
        if p == rank:
            continue
        local = puerta(filas, host, ph, half)
        remota = puerta(filas, ph, host, half)
        entradas[p] = (local, remota)
        locales.add(local)
    hcas = sorted(locales, key=pci)
    if len(hcas) != 2:
        raise ValueError(f"{host}: puertas {half} -> {hcas}; hacen falta dos HCAs distintas (una por vecino)")
    for local, _ in entradas.values():
        if local not in hcas:
            raise ValueError(f"{host}: {local} no esta en B12X_ROCE_HCA={hcas}")
    return rank, hcas, entradas


def lineas_rango(filas, host, half):
    rank, hcas, entradas = mapa_rango(filas, host, half)
    return [
        "B12X_ROCE_RING=1",
        f"B12X_ROCE_HCA={','.join(hcas)}",
        "B12X_ROCE_PEER_HCAS=" + ",".join(
            f"{p}:{local}:{remota}" for p, (local, remota) in sorted(entradas.items())
        ),
    ]


def comprobar(filas, half):
    """Simetria cruzada de los tres rangos: lo que el runtime va a exigir al arrancar."""
    mapas = {}
    for rank, host in enumerate(ANILLO):
        _, hcas, entradas = mapa_rango(filas, host, half)
        mapas[rank] = (hcas, entradas)
    for r, (hcas, entradas) in mapas.items():
        for p, (local, remota) in entradas.items():
            phcas, pentradas = mapas[p]
            pl, pr = pentradas[r]
            if [pl, pr] != [remota, local]:
                raise ValueError(
                    f"asimetria {ANILLO[r]}<->{ANILLO[p]}: {r} dice {local}->{remota}, "
                    f"{p} dice {pl}->{pr}"
                )
            if local not in hcas or remota not in phcas:
                raise ValueError(f"{ANILLO[r]}: {local} o {remota} fuera de las listas de HCA")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--host", help="host de ring.conf de ESTE rango: dgx1, dgx2 o dgx3")
    grupo.add_argument("--all", action="store_true", help="imprime los tres rangos, con cabecera")
    grupo.add_argument("--check", action="store_true", help="solo comprueba la simetria cruzada")
    ap.add_argument("--half", default="A", choices=("A", "B"), help="mitad del par de ring.conf (default A)")
    ap.add_argument("--ring-conf", type=Path, default=RING_CONF)
    a = ap.parse_args(argv)
    try:
        filas = leer_ring_conf(a.ring_conf)
        if a.check:
            comprobar(filas, a.half)
            print(f"SIMETRIA ok ({a.half}) anillo={','.join(ANILLO)}")
            return 0
        if a.all:
            for rank, host in enumerate(ANILLO):
                print(f"# rango {rank} ({host})")
                print(*lineas_rango(filas, host, a.half), sep="\n")
            return 0
        print(*lineas_rango(filas, a.host, a.half), sep="\n")
    except (OSError, ValueError) as e:
        sys.exit(f"roce_ring_map: {e}")


if __name__ == "__main__":
    main()
