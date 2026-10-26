#!/usr/bin/env python3
"""Control lambda=0 bit-exact. Se corre DENTRO de la imagen construida:

  docker run --rm --platform linux/arm64 --entrypoint python3 \
    -v $PWD/test_lambda_zero_identity.py:/tmp/t.py:ro <IMAGEN> /tmp/t.py

No esta cableado como gate del Dockerfile a proposito: el algebra ya la cubre
test_projection_equivalence.py en el Gate 3. Esto comprueba algo distinto y mas
estrecho -- que lambda=0 no mueve NI UN BIT -- y es el control que hay que pasar
antes de mover el dial en produccion.

Va en CPU: la proyeccion es y - lam*r_hat(r_hat.y) y no depende del device.

A) sin VLLM_REFUSAL_DIRECTION: no hay estado y project() devuelve y intacto.
B) con la variable y lambda=0: la resta SI se ejecuta. max|delta| debe ser 0
   EXACTO. Si no lo es, NO se afloja el umbral: es que lambda no llega a 0 (un
   1e-8 en el buffer por token, o fill_neutral rellenando con el global).
C) control negativo: con lambda=1.5 la proyeccion TIENE que mover el tensor. Sin
   este bloque, un project() que no hiciera nada pasaria A y B y seria mentira.
"""
import os
import sys

import numpy as np

NPY = os.environ.get("REFUSAL_NPY", "/opt/refusal/refusal_direction_flashnext.npy")
H = int(np.load(NPY).shape[-1])

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


# ------------------------------------------------------------------ A
os.environ.pop("VLLM_REFUSAL_DIRECTION", None)
os.environ.pop("SGLANG_REFUSAL_DIRECTION", None)

import torch  # noqa: E402
import vllm.refusal_projection as rp  # noqa: E402

check("A1 sin variable no hay estado",
      rp.init_from_env(H) is None and not rp.is_enabled())
for dt in (torch.bfloat16, torch.float16, torch.float32):
    y = torch.randn(64, H, dtype=dt)
    check(f"A2 identidad exacta {dt}", torch.equal(rp.project(y, key="t"), y))

# ------------------------------------------------------------------ B
import importlib  # noqa: E402

os.environ["VLLM_REFUSAL_DIRECTION"] = NPY
os.environ["VLLM_REFUSAL_LAMBDA"] = "0"
rp = importlib.reload(rp)
st = rp.init_from_env(H, device=torch.device("cpu"))
check("B1 con variable hay estado", st is not None and rp.is_enabled())
check("B2 lambda es 0.0 exacto", rp.get_lambda() == 0.0, f"lambda={rp.get_lambda()!r}")

for dt in (torch.bfloat16, torch.float16, torch.float32):
    torch.manual_seed(0)
    y = torch.randn(256, H, dtype=dt)
    ref = y.clone()
    out = rp.project(y, key="t")
    dmax = (out.float() - ref.float()).abs().max().item()
    check(f"B3 max|delta| == 0 {dt}", dmax == 0.0, f"max|delta|={dmax!r}")
    check(f"B4 bit-exact {dt}", torch.equal(out, ref))

# ------------------------------------------------------------------ C
rp.set_lambda(1.5)
torch.manual_seed(0)
y = torch.randn(256, H, dtype=torch.bfloat16)
ref = y.clone()
dmax = (rp.project(y, key="t").float() - ref.float()).abs().max().item()
check("C1 control negativo: lambda=1.5 mueve", dmax > 0.0, f"max|delta|={dmax:.4f}")

rp.set_lambda(0.0)
torch.manual_seed(0)
y = torch.randn(256, H, dtype=torch.bfloat16)
ref = y.clone()
check("C2 volver a 0 restaura la identidad", torch.equal(rp.project(y, key="t"), ref))

print("\nFALLOS:", fails if fails else "ninguno")
sys.exit(1 if fails else 0)
