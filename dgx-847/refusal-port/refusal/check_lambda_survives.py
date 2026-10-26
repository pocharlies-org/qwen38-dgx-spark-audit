#!/usr/bin/env python3
"""Gate 11: el lambda POR PETICION sobrevive la secuencia real del runner.

03-09-2026. `fill_target` escribe una fila de lambda por token. Seis lineas
despues, en el MISMO bloque del prep del target, se llamaba a
`fill_draft_neutral`, que es un alias de `fill_neutral` y hace
`st.tok.fill_(global)` sobre TODO el buffer:

    tras fill_target        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    tras fill_draft_neutral [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

El sello por peticion no tenia NINGUN efecto y el lambda global ganaba siempre.
Lo disimulaba que el dial global si funciona: `set_lambda` escribe el buffer
entero, asi que mover el dial global se notaba y el `cache_salt` no.

Dos comprobaciones, y las dos salen del arbol parcheado -- escribir la secuencia
a mano seria copiar el bug en el test, como ya paso con el gate 9:

  A (estatico) ningun bloque que llame a `fill_target` puede llamar tambien a
    `fill_neutral`/`fill_draft_neutral`: eso es el clobber, por construccion.
  B (estatico) la neutralizacion del drafter tiene que seguir existiendo y estar
    pegada a `speculator.propose`, o el draft hereda el lambda del target.
  C (runtime)  `fill_target` escribe de verdad valores por peticion distintos
    del global. Si eso regresara, A y B pasarian sobre codigo muerto.
"""
import ast
import os
import sys
from pathlib import Path

fails = []


def check(nombre, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {nombre}  {extra}")
    if not cond:
        fails.append(nombre)


def metodos_refusal(nodo):
    """Nombres de metodo de las llamadas self.refusal_state.X(...) bajo `nodo`."""
    out = []
    for n in ast.walk(nodo):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            duenyo = n.func.value
            if (isinstance(duenyo, ast.Attribute) and duenyo.attr == "refusal_state"
                    and isinstance(duenyo.value, ast.Name) and duenyo.value.id == "self"):
                out.append((n.lineno, n.func.attr))
    return sorted(out)


def main(site: str) -> int:
    ruta = Path(site) / "v1" / "worker" / "gpu" / "model_runner.py"
    src = ruta.read_text()
    arbol = ast.parse(src)

    todas = metodos_refusal(arbol)
    check("el runner llama al estado de refusal", bool(todas),
          f"{len(todas)} llamadas")
    if not todas:
        return 1

    # --- A: ningun bloque mezcla fill_target con un fill_* que arrasa el buffer
    CLOBBER = {"fill_neutral", "fill_draft_neutral"}
    malos = []
    for n in ast.walk(arbol):
        if not isinstance(n, (ast.If, ast.With, ast.For, ast.While)):
            continue
        cuerpo = ast.Module(body=n.body, type_ignores=[])
        nombres = {m for _, m in metodos_refusal(cuerpo)}
        if "fill_target" in nombres and (nombres & CLOBBER):
            malos.append((n.lineno, sorted(nombres)))
    check("nada arrasa el buffer tras fill_target", not malos,
          "" if not malos else f"bloque en linea {malos[0][0]}: {malos[0][1]}")

    # --- B: la neutralizacion del drafter existe y esta pegada a propose()
    draft = [ln for ln, m in todas if m == "fill_draft_neutral"]
    check("la neutralizacion del drafter sigue existiendo", bool(draft),
          f"lineas {draft}")
    if draft:
        propose = [n.lineno for n in ast.walk(arbol)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "propose"]
        # Cota 12 (no 8): en el fork del b12x entre la neutralizacion y el
        # `speculator.propose(` real esta el bloque del boundary checkpoint. La
        # regla que importa sigue siendo la misma: el fill del drafter va en la
        # ENTRADA del borrador, no pegado al fill_target del target (eso es lo que
        # rompio el sello por peticion el 03-09-2026, a cientos de lineas).
        cerca = [d for d in draft if any(0 < p - d <= 12 for p in propose)]
        check("y esta pegada a speculator.propose()", bool(cerca),
              f"draft={draft} propose={propose}")

    # --- C: fill_target escribe de verdad valores por peticion
    import numpy as np
    import torch
    os.environ["VLLM_REFUSAL_DIRECTION"] = os.environ.get(
        "REFUSAL_NPY", "/opt/refusal/refusal_direction_flashnext.npy")
    os.environ["VLLM_REFUSAL_LAMBDA"] = "1.0"
    import vllm.refusal_projection as R
    R.init_from_env(2560, device=torch.device("cpu"))
    R.add_request(0, 0.0)                      # la peticion 0 pide refusal:0
    R.fill_target(np.array([0, 1]), np.array([3, 3]), 1.0, num_tokens=6)
    fila = R._STATE.tok[:6].tolist()
    check("fill_target escribe lambda por peticion", fila[:3] == [0.0] * 3,
          f"tok[:6]={fila}")

    print()
    print("FALLOS:", ", ".join(fails) if fails else "ninguno")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
