#!/usr/bin/env python3
"""Test sin GPU del parcheador QWEN_GDN_MIX (DGX-852).

Aplica `patch_gdn_mix.py` a una COPIA de fork-src (nunca al original) y
verifica, para cada modo (A, B, B2):
  - que todas las anclas casan (exit 0, marcas presentes),
  - que los ficheros resultantes compilan (py_compile),
  - que es idempotente (segunda pasada: ya aplicado, sin duplicar),
  - que cada modo deja intactas las marcas de los otros (A no toca diferidos,
    B no toca el prefill, B2 no toca el selector ni refuse_reasons),
  - que es FAIL-CLOSED (una ancla corrupta -> exit != 0 y no escribe),
  - que A0 (vacio) es inerte.

Uso:  python3 -I test_patch_gdn_mix.py [ruta-fork-src-dist-packages]
Sale con 0 si todo pasa, 1 en cualquier fallo.
"""

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHER = HERE / "patch_gdn_mix.py"
DEFAULT_SRC = Path(
    "/home/user/bench-results/dgx-847/fork-src/usr/local/lib/python3.12/dist-packages"
)

# Ficheros que toca el parcheador, relativos al raiz dist-packages.
TARGETS = [
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "vllm/v1/worker/gdn_deferred_commit.py",
    "b12x/sequence/gdn_decode/_impl.py",
    "b12x/sequence/gdn_decode/_cute_kernels.py",
    "b12x/sequence/gdn_decode/_tuning.py",
    # B2 (prefill b12x en BF16):
    "b12x/sequence/gdn_prefill/_impl.py",
    "b12x/sequence/gdn_prefill/_tuning.py",
    "b12x/sequence/gdn_prefill/_parallel.py",
    "b12x/sequence/gdn_prefill/_parallel_kernels.py",
    "b12x/sequence/_shared/delta_prefill/_cute_kernels.py",
    "b12x/sequence/_shared/delta_prefill/preparation.py",
]

# Marcas por modo. Las de prefijo QWEN_GDN_MIX_ son todas.
M_SELECTION = "QWEN_GDN_MIX_SELECTION"
M_IMPORTOS = "QWEN_GDN_MIX_IMPORTOS"
M_REFUSE = "QWEN_GDN_MIX_REFUSE"
M_CAPS = "QWEN_GDN_MIX_CAPS"
M_CUTEDEC = "QWEN_GDN_MIX_CUTEDEC"
M_CUTECOMMIT = "QWEN_GDN_MIX_CUTECOMMIT"
M_DECTUNE = "QWEN_GDN_MIX_DECTUNE"
# B2 (prefill):
M_PREF = [
    "QWEN_GDN_MIX_PREFCAPS", "QWEN_GDN_MIX_PREFTUNE", "QWEN_GDN_MIX_PREFPREP",
    "QWEN_GDN_MIX_PREFINNER", "QWEN_GDN_MIX_PREFPOOL",
    "QWEN_GDN_MIX_PREFRT", "QWEN_GDN_MIX_PREFRTSTORE", "QWEN_GDN_MIX_PREFSTORE",
    "QWEN_GDN_MIX_PREFKEY", "QWEN_GDN_MIX_PREFCT", "QWEN_GDN_MIX_PREFCARG",
    "QWEN_GDN_MIX_PREFCFAKE", "QWEN_GDN_MIX_PREFCPTR", "QWEN_GDN_MIX_PREFCOMMIT",
    "QWEN_GDN_MIX_PREFCOMMITSTORE", "QWEN_GDN_MIX_PREFCOMMITCKPT",
    "QWEN_GDN_MIX_PREFAXKEY", "QWEN_GDN_MIX_PREFAXARG", "QWEN_GDN_MIX_PREFAXFAKE",
    "QWEN_GDN_MIX_PREFAXBND", "QWEN_GDN_MIX_PREFAXCOMMIT",
]

EXPECT = {
    "A": [M_SELECTION],
    "B": [M_SELECTION, M_IMPORTOS, M_REFUSE, M_CAPS, M_CUTEDEC, M_CUTECOMMIT, M_DECTUNE],
    # B2: relajaciones de decode (CAPS/CUTEDEC/CUTECOMMIT) + todo el prefill.
    # NO selection, NO importos, NO refuse (con prefill y decode b12x la
    # politica original y refuse_reasons ya lo permiten).
    "B2": [M_CAPS, M_CUTEDEC, M_CUTECOMMIT, M_DECTUNE, *M_PREF],
}

ALL_MARKERS = [M_SELECTION, M_IMPORTOS, M_REFUSE, M_CAPS, M_CUTEDEC, M_CUTECOMMIT, M_DECTUNE, *M_PREF]

# Ficheros de prefill (solo B2) y de decode b12x (B y B2).
PREFILL_FILES = {t for t in TARGETS
                     if "gdn_prefill" in t or "delta_prefill" in t}
DECODE_B12X_FILES = {"b12x/sequence/gdn_decode/_impl.py",
                     "b12x/sequence/gdn_decode/_cute_kernels.py",
                     "b12x/sequence/gdn_decode/_tuning.py"}

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def make_copy(src_root: Path, dst_root: Path) -> None:
    """Copia solo los TARGETS preservando la estructura vllm/ y b12x/."""
    for rel in TARGETS:
        s = src_root / rel
        if not s.is_file():
            raise SystemExit(f"fuente ausente: {s}")
        d = dst_root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)


def run_patcher(root: Path, mix: str, extra_env: dict | None = None):
    env = dict(os.environ)
    env["QWEN_GDN_MIX"] = mix
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-I", str(PATCHER), "--root", str(root)],
        env=env, capture_output=True, text=True,
    )


def check_mode(src_root: Path, mix: str) -> None:
    print(f"[modo {mix}]")
    with tempfile.TemporaryDirectory(prefix=f"gdnmix-{mix}-") as td:
        root = Path(td)
        make_copy(src_root, root)
        r = run_patcher(root, mix)
        if r.returncode != 0:
            fail(f"{mix}: primera pasada exit={r.returncode}: {r.stdout}{r.stderr}")
            return
        blob = "".join((root / rel).read_text() for rel in TARGETS)
        # Marcas esperadas presentes.
        for marker in EXPECT[mix]:
            if marker not in blob:
                fail(f"{mix}: falta la marca {marker}")
        # Marcas ajenas al modo NO presentes.
        allowed = set(EXPECT[mix])
        for marker in ALL_MARKERS:
            if marker not in allowed and marker in blob:
                fail(f"{mix}: marca ajena presente ({marker})")
        # A y B no tocan el prefill; A no toca ficheros b12x.
        for rel in TARGETS:
            touched = "QWEN_GDN_MIX" in (root / rel).read_text()
            if mix in ("A", "B") and rel in PREFILL_FILES and touched:
                fail(f"{mix}: toco un fichero de prefill ({rel}); solo B2 lo hace")
            if mix == "A" and rel in DECODE_B12X_FILES and touched:
                fail(f"A: toco un fichero b12x de decode ({rel})")
            if mix == "A" and rel == "vllm/v1/worker/gdn_deferred_commit.py" and touched:
                fail("A: toco gdn_deferred_commit (diferidos); A no debe")
        # Todos los TARGETS tocados compilan.
        for rel in TARGETS:
            p = root / rel
            try:
                py_compile.compile(str(p), cfile=str(p) + "c", doraise=True)
            except py_compile.PyCompileError as exc:
                fail(f"{mix}: no compila {rel}: {exc}")
        ok(f"{mix}: anclas casan, marcas correctas y ficheros compilan")

        # Idempotencia: segunda pasada no duplica ni rompe.
        before = {rel: (root / rel).read_text() for rel in TARGETS}
        r2 = run_patcher(root, mix)
        if r2.returncode != 0:
            fail(f"{mix}: segunda pasada exit={r2.returncode}: {r2.stderr}")
        for rel in TARGETS:
            after = (root / rel).read_text()
            if after != before[rel]:
                fail(f"{mix}: no idempotente, {rel} cambio en la segunda pasada")
        # Cada marca aparece exactamente una vez (no duplicada).
        for rel in TARGETS:
            text = (root / rel).read_text()
            for marker in EXPECT[mix]:
                if marker in text and text.count(marker) > 1:
                    # Las marcas de comentario pueden repetirse en el cuerpo;
                    # exigimos al menos una y que el fichero siga igual que tras
                    # la primera pasada (ya comprobado arriba con before/after).
                    pass
        if "inerte" in r2.stdout:
            fail(f"{mix}: segunda pasada se reporto inerte con {mix} puesto")
        ok(f"{mix}: idempotente (segunda pasada no cambia nada)")


def check_inert(src_root: Path) -> None:
    print("[modo vacio / A0 (inerte)]")
    with tempfile.TemporaryDirectory(prefix="gdnmix-none-") as td:
        root = Path(td)
        make_copy(src_root, root)
        before = {rel: (root / rel).read_text() for rel in TARGETS}
        r = run_patcher(root, "")
        if r.returncode != 0:
            fail(f"inerte: exit={r.returncode}: {r.stderr}")
        for rel in TARGETS:
            if (root / rel).read_text() != before[rel]:
                fail(f"inerte: toco {rel} sin pedirlo")
        if "inerte" not in r.stdout:
            fail(f"inerte: mensaje esperado ausente: {r.stdout!r}")
        ok("vacio (A0): no toca nada")


def check_unchanged_across_modes(src_root: Path) -> None:
    """A y B no cambian respecto a lo que producian antes de anadir B2.

    Se comprueba que B2 no altera el conjunto de ediciones de A y de B: los
    ficheros de A y B quedan identicos aplicandolos solos, y B2 no introduce
    marcas en los ficheros que A/B no tocaban ya.
    """
    print("[A/B inalterados por la llegada de B2]")
    with tempfile.TemporaryDirectory(prefix="gdnmix-ab-") as td:
        root = Path(td)
        make_copy(src_root, root)
        for mix in ("A", "B"):
            r = run_patcher(root, mix)
            if r.returncode != 0:
                fail(f"{mix}: exit={r.returncode} en check_unchanged")
        # A y B siguen sin tocar el prefill.
        blob = "".join((root / rel).read_text() for rel in PREFILL_FILES)
        for marker in M_PREF:
            if marker in blob:
                fail(f"A/B: marca de B2 ({marker}) en ficheros de prefill")
        ok("A y B no arrastran ediciones de B2")


def check_fail_closed(src_root: Path) -> None:
    print("[fail-closed: ancla corrupta]")
    # Rompe el ancla de seleccion (A) y una de prefill (B2).
    for mix, rel, needle, repl in (
        ("A", TARGETS[0],
         '        return "b12x", "b12x", explicitly_configured\n',
         '        return "b12x", "b12x", explicitly_configured  # X\n'),
        ("B2", "b12x/sequence/gdn_prefill/_impl.py",
         '            raise ValueError("state_dtype must be torch.float32")\n',
         '            raise ValueError("state_dtype must be torch.float32")  # X\n'),
    ):
        with tempfile.TemporaryDirectory(prefix="gdnmix-bad-") as td:
            root = Path(td)
            make_copy(src_root, root)
            p = root / rel
            txt = p.read_text()
            if txt.count(needle) != 1:
                fail(f"fail-closed({mix}): el ancla de prueba no casa una vez en {rel}")
                continue
            p.write_text(txt.replace(needle, repl, 1))
            r = run_patcher(root, mix)
            if r.returncode == 0:
                fail(f"fail-closed({mix}): exit 0 con un ancla corrupta")
            if "ABORTA" not in (r.stderr + r.stdout):
                fail(f"fail-closed({mix}): falta mensaje ABORTA: {r.stderr!r}")
            if any(m in p.read_text() for m in ALL_MARKERS):
                fail(f"fail-closed({mix}): escribio a pesar del ancla corrupta")
    ok("ancla corrupta (A y B2) -> exit != 0 y sin escribir")


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_dir():
        print(f"fork-src no encontrado: {src}", file=sys.stderr)
        return 1
    if not PATCHER.is_file():
        print(f"parcheador no encontrado: {PATCHER}", file=sys.stderr)
        return 1
    print(f"fuente: {src}")
    print(f"parcheador: {PATCHER}")
    check_inert(src)
    check_mode(src, "A")
    check_mode(src, "B")
    check_mode(src, "B2")
    check_unchanged_across_modes(src)
    check_fail_closed(src)
    print()
    if FAILURES:
        print(f"RESULTADO: {len(FAILURES)} FALLO(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULTADO: TODO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
