#!/usr/bin/env python3
"""La equivalencia en la que se apoya todo el parche, comprobada en numeros.

Proyectar la SUMA de las salidas antes de inyectarla == ortogonalizar cada writer
que la suma. Si esto fallase, los cuatro sitios del parche estarian mal elegidos.
Corre en CPU, sin GPU y sin sglang.
"""
import sys, types, numpy as np, torch

sys.path.insert(0, "payload")
import refusal_projection as R

H, IN, N = 64, 40, 7
LAM = 1.5
g = torch.Generator().manual_seed(0)
v = torch.randn(H, generator=g); v /= v.norm()

R._STATE = R._State(v, LAM, torch.device("cpu"), H)

def edit(W):                      # el algebra del autor: W - lam * v (v^T W)
    return W - LAM * torch.outer(v, v @ W)

fails = 0

# 1) un writer suelto
W = torch.randn(H, IN, generator=g); x = torch.randn(N, IN, generator=g)
lhs = x @ edit(W).T
rhs = R.project(x @ W.T)
e1 = (lhs - rhs).abs().max().item()
print(f"1) un writer                       err_max={e1:.3e}")
fails += e1 > 1e-5

# 2) suma de writers (o_proj + shared + expertos) proyectada UNA vez
Ws = [torch.randn(H, IN, generator=g) for _ in range(5)]
xs = [torch.randn(N, IN, generator=g) for _ in range(5)]
lhs = sum(xi @ edit(Wi).T for Wi, xi in zip(Ws, xs))
rhs = R.project(sum(xi @ Wi.T for Wi, xi in zip(Ws, xs)))
e2 = (lhs - rhs).abs().max().item()
print(f"2) suma de 5 writers, 1 proyeccion err_max={e2:.3e}")
fails += e2 > 1e-5

# 3) embed_tokens: project_in sobre las FILAS == proyectar la salida por token
E = torch.randn(200, H, generator=g)
E_ed = E - LAM * (E @ v).unsqueeze(-1) * v
ids = torch.tensor([3, 17, 199, 0])
e3 = (E_ed[ids] - R.project(E[ids])).abs().max().item()
print(f"3) embed_tokens (project_in)       err_max={e3:.3e}")
fails += e3 > 1e-5

# 4) tronco hiper-conectado: por stream
HC = 4
y = torch.randn(N, HC * H, generator=g)
ref = (y.reshape(N, HC, H) - LAM * ((y.reshape(N, HC, H) @ v).unsqueeze(-1) * v)).reshape(N, HC * H)
e4 = (ref - R.project_hc(y, HC)).abs().max().item()
print(f"4) apply_hc por stream             err_max={e4:.3e}")
fails += e4 > 1e-5

# 5) lambda=0 es la identidad EXACTA (bit a bit)
R.set_lambda(0.0)
y = torch.randn(N, H, generator=g)
same = torch.equal(y, R.project(y))
print(f"5) lambda=0 identidad bit a bit    {same}")
fails += not same

# 6) el dial muta el MISMO tensor (lo que hace que el CUDA graph lo vea)
ptr = R._STATE.lam.data_ptr()
R.set_lambda(1.5)
print(f"6) set_lambda muta in-place        {R._STATE.lam.data_ptr() == ptr}")
fails += R._STATE.lam.data_ptr() != ptr

# 7) fail-closed ante una forma que no es el hidden completo
try:
    R.project(torch.randn(N, H - 1)); ok = False
except RuntimeError:
    ok = True
print(f"7) forma incorrecta -> RuntimeError {ok}")
fails += not ok

print("FAIL" if fails else "\nTODO OK")
sys.exit(1 if fails else 0)
