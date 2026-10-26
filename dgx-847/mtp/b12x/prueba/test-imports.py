#!/usr/bin/env python3
"""DGX-852: que exige importar b12x, contra lo que trae la imagen del residente.

Recorre TODOS los .py de la rueda (no solo los seis submodulos que importa
`vllm/utils/b12x.py:33-42`: esos cargan su api de forma perezosa por nombre, y el ast no lo
ve) y lista los modulos de terceros que se importan, para comprobarlos contra las
distribuciones instaladas en la imagen.

    python3 -I test-imports.py <dir-desempaquetado> <imagen-distpackages.txt>
"""

import ast
import sys
from pathlib import Path

# modulo de import -> distribucion de PyPI que lo trae
DISTRIBUCION = {
    "torch": "torch",
    "triton": "triton",
    "cutlass": "nvidia-cutlass-dsl",
    "cuda": "cuda-python",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "rich": "rich",
    "numpy": "numpy",
    "tvm_ffi": "apache-tvm-ffi",
    "psutil": "psutil",
    "packaging": "packaging",
    "yaml": "pyyaml",
}


def raices(paquete: Path) -> dict[str, list[str]]:
    """modulo de raiz de terceros -> ficheros que lo importan."""
    out: dict[str, list[str]] = {}
    stdlib = sys.stdlib_module_names
    for fichero in paquete.rglob("*.py"):
        try:
            arbol = ast.parse(fichero.read_text(), filename=str(fichero))
        except SyntaxError as e:
            print(f"AVISO: no se parsea {fichero.name}: {e}")
            continue
        nombres: list[str] = []
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres += [n.name for n in nodo.names]
            elif isinstance(nodo, ast.ImportFrom) and not nodo.level and nodo.module:
                nombres.append(nodo.module)
        for nombre in nombres:
            raiz = nombre.split(".")[0]
            if raiz in stdlib or raiz == paquete.name:
                continue
            out.setdefault(raiz, []).append(fichero.relative_to(paquete.parent).as_posix())
    return out


def main() -> int:
    paquete = Path(sys.argv[1]) / "b12x"
    listado = {l.strip() for l in Path(sys.argv[2]).read_text().splitlines()}
    total = len(list(paquete.rglob("*.py")))
    exigidos = raices(paquete)
    print(f"{total} ficheros .py del paquete b12x -> {len(exigidos)} modulos de terceros\n")
    mal = []
    for raiz, donde in sorted(exigidos.items()):
        dist = DISTRIBUCION.get(raiz)
        if dist is None:
            print(f"  ?       import {raiz:<14} (sin mapa a distribucion; {len(donde)} ficheros)")
            mal.append(raiz)
            continue
        presente = any(l.startswith(dist.lower().replace("-", "_") + "-") or
                       l.startswith(dist + "-") for l in listado)
        print(f"  {'ok   ' if presente else 'FALTA'} import {raiz:<14} <- {dist:<18} "
              f"({len(donde)} ficheros){'' if presente else '  ej: ' + donde[0]}")
        if not presente:
            mal.append(raiz)
    print(f"\n{'TODO PRESENTE' if not mal else 'FALTAN: ' + ', '.join(sorted(set(mal)))}")
    return 0 if not mal else 1


if __name__ == "__main__":
    sys.exit(main())
