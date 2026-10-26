#!/usr/bin/env python3
"""Gate 9: las cuentas del guard casan con los sitios que el parche registra.

El guard es fail-closed: si `verify_consumed` espera un numero distinto del que
el propio parche produce, el arranque muere. Eso ya paso dos veces, y las dos se
descubrio con el residente caido y 25 minutos de carga de pesos por intento.

Este gate lo resuelve en un segundo y sin GPU, porque el conteo no depende de
pesos ni de CUDA: es un conjunto de cadenas y un patron.

Lo importante del diseno: los DOS lados salen del arbol parcheado.
  - las claves, de las llamadas `_refusal.project*(key=...)`
  - las expectativas, de las llamadas `_refusal.verify_consumed(...)`
Escribir aqui los numeros a mano seria copiar el bug en el test.

Se evalua con varias topologias, no solo la de produccion: una formula erronea
puede acertar por casualidad con un unico juego de valores.
"""
import ast
import re
import sys
from pathlib import Path

# Config real, medida en el pod (config.json del checkpoint NVFP4), mas una
# topologia pequena de control con dos capas ple.
TOPOLOGIAS = [
    ("produccion", 48, [2]),
    ("control", 4, [1, 3]),
]
MP = "language_model.model"  # prefijo del modelo; cualquiera vale, es simetrico


def _fstring_a_patron(expr: str, prefijo: str) -> str:
    """f'{self._rank1_prefix}.attn_out.{self.layer_type}' -> '<prefijo>.attn_out.X'.

    Los campos dinamicos que no son el prefijo se sustituyen por un token sin
    puntos: su valor no altera el tipo de sitio, y _site_kind lo determina por
    componentes.
    """
    cuerpo = re.sub(r"^f['\"]|['\"]$", "", expr)
    cuerpo = cuerpo.replace("{self._rank1_prefix}", prefijo).replace("{_prefix}", prefijo)
    return re.sub(r"\{[^}]*\}", "X", cuerpo)


def main(site: str) -> int:
    ruta = Path(site) / "models" / "qwen3_8_flash_next" / "model.py"
    arbol = ast.parse(ruta.read_text())

    # ---- lado A: que claves registra el parche, y desde que ambito
    claves_capa: list[str] = []
    claves_modelo: list[str] = []
    verifica: list[tuple[str, str, str]] = []

    for cls in [n for n in ast.walk(arbol) if isinstance(n, ast.ClassDef)]:
        es_capa = "DecoderLayer" in cls.name
        for n in ast.walk(cls):
            if not isinstance(n, ast.Call):
                continue
            f = ast.unparse(n.func)
            if f in ("_refusal.project", "_refusal.project_hc"):
                kw = [k for k in n.keywords if k.arg == "key"]
                if kw:
                    (claves_capa if es_capa else claves_modelo).append(ast.unparse(kw[0].value))
            elif f == "_refusal.verify_consumed":
                sfx = [k for k in n.keywords if k.arg == "suffix"]
                verifica.append((
                    ast.unparse(n.args[0]),
                    ast.unparse(n.args[1]),
                    ast.literal_eval(sfx[0].value) if sfx else "",
                ))

    if not claves_capa or not verifica:
        print("FAIL no encuentro las llamadas del parche: el ancla ha cambiado")
        return 1

    sys.path.insert(0, site.rsplit("/vllm", 1)[0] if "/vllm" in site else site)
    from vllm.refusal_projection import _site_kind  # noqa: E402

    fallos = 0
    for nombre, n_capas, ple_ids in TOPOLOGIAS:
        consumido: set[str] = set()
        for i in range(n_capas):
            lp = f"{MP}.layers.{i}"
            for expr in claves_capa:
                patron = _fstring_a_patron(expr, lp)
                # el ple solo existe en las capas que lo declaran
                if patron.endswith(".ple") and i not in ple_ids:
                    continue
                consumido.add(patron)
        for expr in claves_modelo:
            consumido.add(_fstring_a_patron(expr, MP))

        entorno = {
            "_prefix": MP,
            "_n": n_capas,
            "_ple": len({abs(i) for i in ple_ids if abs(i) <= n_capas}),
            "config": type("C", (), {"num_hidden_layers": n_capas})(),
        }
        print(f"--- {nombre}: {n_capas} capas, ple en {ple_ids} "
              f"({len(consumido)} sitios registrados)")
        for pref_expr, exp_expr, suffix in verifica:
            prefijo = eval(pref_expr, {}, entorno)      # noqa: S307 - expr del arbol
            esperado = eval(exp_expr, {}, entorno)      # noqa: S307
            obtenido = sum(1 for k in consumido
                           if k.startswith(prefijo) and _site_kind(k) == suffix)
            ok = obtenido == esperado
            fallos += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} verify_consumed({prefijo!r}, "
                  f"{esperado}, suffix={suffix!r}) -> registrados={obtenido}")

    if fallos:
        print(f"\ngate de cuentas: {fallos} verificacion(es) del guard NO casan "
              f"con los sitios del parche. El arranque moriria.")
        return 1
    print(f"\ngate de cuentas OK: las {len(verifica)} verificaciones casan en "
          f"{len(TOPOLOGIAS)} topologias")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
