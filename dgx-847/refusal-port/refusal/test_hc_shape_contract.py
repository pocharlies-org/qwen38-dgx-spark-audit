#!/usr/bin/env python3
"""Gate 10: el contrato de forma del espacio hc.

`project_hc` recibe el tensor en espacio hc APLANADO -- ultimo eje hc*hidden --
y proyecta cada stream por separado. Los dos sitios que lo usan (el PLE del
target y fc_embedding+fc_hidden del drafter MTP) tienen que pasarlo asi.

03-09-2026: el sitio del drafter lo pasaba SEPARADO, [tokens, hc, hidden], y
tumbo el arranque en el primer forward:

    rank1-refusal(hc): mtp.fc_embedding+fc_hidden esperaba ultimo eje 10240,
    visto (4096, 4, 2560)

Los gates 8 (nombres) y 9 (cuentas) no ven formas. Este fija el contrato en CPU
-- la proyeccion no depende del device -- y comprueba tres cosas: que la forma
separada se RECHAZA en vez de colar, que aplanar-y-proyectar es exactamente la
proyeccion por stream, y que no es un no-op disfrazado.
"""
import os
import sys

import numpy as np
import torch

fails = []


def check(nombre, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {nombre}  {extra}")
    if not cond:
        fails.append(nombre)


# REFUSAL_NPY permite correr el contrato en CPU fuera del pod (sequia del
# instalador); dentro del pod no se pone y vale la ruta instalada.
DIR = os.environ.get("REFUSAL_NPY", "/opt/refusal/refusal_direction_flashnext.npy")
os.environ["VLLM_REFUSAL_DIRECTION"] = DIR
os.environ["VLLM_REFUSAL_LAMBDA"] = "1.0"

import vllm.refusal_projection as R  # noqa: E402

H, HC, N = 2560, 4, 37
R.init_from_env(H, device=torch.device("cpu"))
check("hay estado (si no, esto no probaria nada)", R._STATE is not None)

torch.manual_seed(0)
x = torch.randn(N, HC, H, dtype=torch.float32)

# 1. la forma separada NO debe colar: si colase, la proyeccion se aplicaria
#    sobre el eje equivocado y el modelo saldria mal ablado sin avisar.
try:
    R.project_hc(x, HC, key="separado")
    check("la forma separada se rechaza", False, "colo, y no deberia")
except RuntimeError as e:
    check("la forma separada se rechaza", "esperaba ultimo eje" in str(e),
          f"-> {str(e)[:60]}")

# 2. aplanar y proyectar == proyectar cada stream por separado
y = R.project_hc(x.flatten(-2), HC, key="aplanado").reshape(N, HC, H)
d = torch.from_numpy(np.load(DIR)).float()
d = d / d.norm()
ref = x - 1.0 * (x @ d).unsqueeze(-1) * d
err = float((y - ref).abs().max())
check("aplanar+project_hc == proyeccion por stream", err < 1e-4, f"max|delta|={err:.2e}")

# 3. y no es un no-op: con lambda=1 tiene que mover el tensor
mov = float((x - y).abs().max())
check("no es un no-op", mov > 1e-3, f"max|x-y|={mov:.4f}")

print()
print("FALLOS:", ", ".join(fails) if fails else "ninguno")
sys.exit(1 if fails else 0)
