#!/usr/bin/env python3
"""El lambda POR PETICION, en el mismo lote.

Lo que se comprueba, en el orden en que puede romperse:
  1. Dos peticiones con lambdas distintos en UN solo lote: cada tramo de tokens
     sale con el suyo. Si el buffer estuviera indexado al reves, esto pasa igual
     y no se detecta -- por eso se miran AMBOS tramos, no solo el corregido.
  2. Un sello `refusal:0` es una peticion que PIDE la base. Confundirla con
     "sin sello" le dejaria el lambda global, o sea el de otra peticion.
  3. Un slot de peticion reusado sin sello vuelve al global. Si el registro
     guardase el ultimo lambda del slot, la peticion nueva heredaria el anterior.
  4. Quien no trae sello usa el global que pasa el caller, no un valor guardado.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "payload"))
import refusal_projection as R  # noqa: E402

H = 2560
fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")


def main():
    st = R.init_from_env(H, device=torch.device("cpu"))
    if st is None:
        raise SystemExit("falta VLLM_REFUSAL_DIRECTION")
    R.set_lambda(0.0)
    torch.manual_seed(0)
    y = torch.randn(4, H, dtype=torch.bfloat16)

    # 1 -- lote mixto: peticion 0 a 0.0, peticion 1 a 1.5, 2 tokens cada una
    R.add_request(0, 0.0)
    R.add_request(1, 1.5)
    R.fill_target(np.array([0, 1]), np.array([2, 2]), 0.0)
    check("buffer por token: [0,0,1.5,1.5]",
          [round(float(x), 3) for x in st.tok[:4]] == [0.0, 0.0, 1.5, 1.5],
          f"-> {[round(float(x),2) for x in st.tok[:4]]}")
    out = R.project(y, key="t")
    check("tramo lam=0 sale bit-exacto", torch.equal(out[:2], y[:2]))
    check("tramo lam=1.5 sale corregido", not torch.equal(out[2:], y[2:]))
    # lam=1.5 NO anula la direccion: `y - 1.5 v (v.y)` la INVERTE y la deja al
    # 50% (1-lam = -0.5). Pedir cos~0 aqui seria exigir lam=1. Medido: el cos de
    # salida con lam=1.5 es 1.04e-02 en float64 y EXACTAMENTE el mismo numero que
    # da la referencia algebraica pura, o sea que el payload no introduce error.
    # Lo que se comprueba entonces es que el tramo corregido coincide con la
    # algebra del autor, no que la componente desaparezca.
    ref = y[2:].double() - 1.5 * st.dir32.double() * (
        (y[2:].double() @ st.dir32.double()).unsqueeze(-1)
    )
    err = (out[2:].double() - ref).abs().max().item()
    check("tramo lam=1.5 == algebra del autor (W-lam*v(vW))",
          err < 5e-2, f"err_max={err:.3e} (suelo bf16 sobre |y|~1)")

    # 2 -- sello 0.0 explicito con global 1.5: debe dar la base, no el global
    R.set_lambda(1.5)
    R.fill_target(np.array([0]), np.array([2]), 1.5)
    check("sello 0.0 explicito != ausencia (pide base)",
          torch.equal(R.project(y[:2].clone(), key="z"), y[:2]))

    # 3 -- slot reusado sin sello: vuelve al global
    R.remove_request(1)
    R.set_lambda(0.0)
    R.fill_target(np.array([1]), np.array([2]), 0.0)
    check("slot reusado sin sello usa el global",
          torch.equal(R.project(y[:2].clone(), key="r"), y[:2]))

    # 4 -- el global manda sobre quien no trae sello
    R.set_lambda(1.5)
    R.fill_target(np.array([1]), np.array([2]), 1.5)
    check("global 1.5 mueve a quien no trae sello",
          not torch.equal(R.project(y[:2].clone(), key="g"), y[:2]))

    # parseo del sello
    check("sello con basura alrededor", R.parse_request_lambda("xyzrefusal:2.0a") == 2.0)
    check("sello fuera de cota -> None", R.parse_request_lambda("refusal:99") is None)
    check("sin sello -> None", R.parse_request_lambda(None) is None)

    print("TODO OK" if fails == 0 else f"{fails} FALLOS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
