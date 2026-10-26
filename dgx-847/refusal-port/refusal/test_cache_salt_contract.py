#!/usr/bin/env python3
"""Contrato del sello: lo que litellm PUBLICA, esto CONSUME.

El productor esta contratado al otro lado: en k8s-litellm-pocharlies,
tests/test_uncensored_alias_contract.py fija
    "qwen38-flash-next-uncensored": "refusal:1.0"
y asevera que la entrada manda extra_body {"cache_salt": <ese literal>}.

Aqui se contrata el lado consumidor, que no tenia guardia. Si alguien renombra
el prefijo `refusal:` o cambia el formato en vLLM, el alias -uncensored deja de
ablar Y NADA FALLA: la peticion sale, el modelo responde, y responde censurado.
Es el mismo modo de fallo silencioso que ya mordio tres veces en este port.

Se corre como gate del Dockerfile: es CPU pura y no necesita ni modelo ni GPU.
"""
import sys

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


from vllm.v1.core.sched.output import parse_refusal_lambda as P  # noqa: E402

# El literal exacto del manifiesto de litellm. Hardcodeado a proposito: si
# cambia alli, tiene que cambiarse AQUI a mano, y ese es justo el punto.
# 03-09-2026: 1.5 -> 1.0 al cambiar la direccion a la de orcarouter, que usa la
# proyeccion canonica lambda=1 y es la unica con evaluacion publicada.
check("literal 'refusal:1.0' -> 1.0", P("refusal:1.0") == 1.0, f"-> {P('refusal:1.0')!r}")
# El parser no esta atado a un valor concreto: 1.5 era el sello anterior y sigue
# siendo valido, asi que un cambio de punto de trabajo no exige tocar vLLM.
check("otro valor 'refusal:1.5' -> 1.5", P("refusal:1.5") == 1.5, f"-> {P('refusal:1.5')!r}")

# `refusal:0` pide la base EXPLICITAMENTE y no es lo mismo que no pedir nada:
# sin sello manda el lambda global, que puede no ser 0.
check("'refusal:0' -> 0.0 y no None", P("refusal:0") == 0.0, f"-> {P('refusal:0')!r}")
check("sin sello -> None (cae al global)", P(None) is None)

# Un cache_salt de otro para nada debe ablar ni reventar: el sello convive con
# el uso normal de cache_salt como namespace de prefix cache.
for s in ("otra-cosa", "prefix-cache-v2", "refusal", "refusal:"):
    check(f"salt ajeno {s!r} -> None", P(s) is None, f"-> {P(s)!r}")

# Fuera de rango NO se aplica en silencio: se avisa y se cae al global.
for s in ("refusal:99", "refusal:-99", "refusal:abc"):
    r = P(s)
    check(f"{s!r} no cuela", r is None or -1.0 <= r <= 4.0, f"-> {r!r}")

# El dial en caliente viaja por collective_rpc, que invoca METODOS del worker
# por nombre. Si dejan de ser metodos de la clase, el fan-out a los dos ranks
# falla en tiempo de peticion, no al arrancar.
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402

for m in ("set_refusal_lambda", "get_refusal_lambda"):
    check(f"Worker.{m} alcanzable por collective_rpc", callable(getattr(Worker, m, None)))

print("\nFALLOS:", fails if fails else "ninguno")
sys.exit(1 if fails else 0)
