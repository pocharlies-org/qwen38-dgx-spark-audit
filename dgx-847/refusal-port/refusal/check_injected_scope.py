#!/usr/bin/env python3
"""Gate 8: que el codigo INYECTADO pueda ejecutarse, no solo parsear.

    python3 check_injected_scope.py <SITE>

El 03-09-2026 el arranque murio con

    model.py:333, in forward
        attn_out, key=f"{prefix}.attn_out.{self.layer_type}"
    NameError: name 'prefix' is not defined

`prefix` es parametro del __init__ de la capa y el sitio se nombra en forward,
donde ese nombre no existe. Habia CUATRO asi; solo salto el primero que se
ejecuta, asi que arreglarlos de uno en uno costaba un despliegue por bug.

Los otros siete gates comprueban que el parche APLICA: que el ancla casa, que el
fichero parsea, que los imports resuelven, que el algebra sale. Ninguno mira si
el codigo inyectado tiene sus nombres ligados en el ambito donde acaba viviendo.
Eso es analisis estatico puro y no necesita ni pesos ni GPU.

ALCANCE, a proposito estrecho: solo se revisan las sentencias que tocan
`_refusal`. Son las nuestras. Asi el gate no puede dar falsos positivos sobre
codigo de vLLM, que es lo que convierte un gate en ruido que todos ignoran.
"""
import ast
import builtins
import sys
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def _bound_by(node) -> set:
    """Nombres que LIGA este ambito, y solo este.

    ast.walk es recursivo y no distingue ambitos: a nivel de modulo recogia los
    argumentos de TODAS las funciones del fichero, asi que `prefix` figuraba como
    ligado y el gate aprobaba el codigo que precisamente venia a cazar. Aqui se
    para en cada frontera de ambito (def/class/lambda): de ellas solo se liga el
    nombre, no lo de dentro.
    """
    out = set()

    def collect(n) -> None:
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)

    def walk(n) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                out.add(child.name)   # el nombre si; su interior es otro ambito
                continue
            if isinstance(child, ast.Lambda):
                continue
            collect(child)
            walk(child)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            out.add(arg.arg)
        if a.vararg:
            out.add(a.vararg.arg)
        if a.kwarg:
            out.add(a.kwarg.arg)
    walk(node)
    return out

def _touches_refusal(stmt) -> bool:
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and n.id in ("_refusal", "refusal_projection"):
            return True
    return False


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _names_here(node):
    """Name(Load) de ESTE ambito: no entra en def/class/lambda anidados.

    Sin esto, los parametros del hook anidado (_mod, _n, _prefix) se juzgaban
    contra el ambito de __init__ y salian como no ligados: ruido puro.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            yield n
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, SCOPES):
                continue
            stack.append(ch)


def _nested_scopes(node):
    """def/class anidados de primer nivel dentro de node."""
    stack = [node]
    while stack:
        n = stack.pop()
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield ch
            else:
                stack.append(ch)


def check_file(path: Path):
    """Devuelve [(linea, nombre, funcion)] de nombres libres sin ligar."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [(getattr(e, "lineno", 0), f"SyntaxError: {e.msg}", "-")]

    bad = []

    def scan(node, scopes, fname):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Una funcion anidada ve el ambito de fuera; una clase no crea
            # ambito para lo que se define dentro, asi que no se apila.
            scopes = scopes + [_bound_by(node)]
            fname = node.name
        visible = set().union(*scopes)
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scan(stmt, scopes, fname)
                continue
            if _touches_refusal(stmt):
                for nm in _names_here(stmt):
                    if nm.id not in visible:
                        bad.append((nm.lineno, nm.id, fname))
            for sub in _nested_scopes(stmt):
                scan(sub, scopes, fname)

    scan(tree, [_bound_by(tree) | BUILTINS], "<modulo>")
    return bad


def main() -> int:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else
                open("/opt/refusal/SITE").read().strip())
    targets = []
    for f in site.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if "_refusal." in src:
            targets.append(f)

    if not targets:
        print("FAIL: ningun fichero con codigo inyectado; el parche no se aplico")
        return 1

    total = 0
    for f in sorted(targets):
        bad = check_file(f)
        rel = f.relative_to(site)
        if bad:
            total += len(bad)
            for ln, name, fn in bad:
                print(f"FAIL {rel}:{ln} en {fn}(): nombre libre sin ligar: {name!r}")
        else:
            print(f"ok   {rel}")
    if total:
        print(f"\ngate de ambito: {total} nombre(s) sin ligar en codigo inyectado")
        return 1
    print(f"\ngate de ambito OK: {len(targets)} ficheros, todos los nombres ligados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
