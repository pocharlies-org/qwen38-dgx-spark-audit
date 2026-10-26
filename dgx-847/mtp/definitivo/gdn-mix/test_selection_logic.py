#!/usr/bin/env python3
"""Test de logica del selector parcheado (DGX-852, sin GPU).

Extrae `_resolve_gdn_backend_selection` del fichero YA parcheado (copia de
fork-src) y la ejecuta con stubs minimos para comprobar el comportamiento real,
no solo que compila:

  A/B con prefill=triton + decode=b12x  -> ("triton", "b12x")   (la mezcla)
  sin QWEN_GDN_MIX, misma entrada        -> ValueError          (la politica original)
  prefill=auto + decode=b12x             -> ("b12x", "b12x")    (los dos b12x, intacto)
  prefill=triton + decode=triton         -> ("triton", "triton")(E2, intacto)

Uso: python3 -I test_selection_logic.py [dist-packages-src]
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHER = HERE / "patch_gdn_mix.py"
DEFAULT_SRC = Path(
    "/home/user/bench-results/dgx-847/fork-src/usr/local/lib/python3.12/dist-packages"
)
REL = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
TARGETS = [
    REL,
    "vllm/v1/worker/gdn_deferred_commit.py",
    "b12x/sequence/gdn_decode/_impl.py",
    "b12x/sequence/gdn_decode/_cute_kernels.py",
]

FAILURES: list[str] = []


def fail(m: str) -> None:
    FAILURES.append(m)
    print(f"  FAIL: {m}")


def ok(m: str) -> None:
    print(f"  ok: {m}")


def patch_copy(src_root: Path, mix: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"sel-{mix}-"))
    for rel in TARGETS:
        s = src_root / rel
        d = root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
    env = dict(os.environ)
    env["QWEN_GDN_MIX"] = mix
    r = subprocess.run([sys.executable, "-I", str(PATCHER), "--root", str(root)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return root


def load_selector(root: Path):
    """Devuelve la funcion `_resolve_gdn_backend_selection` ejecutada con stubs."""
    text = (root / REL).read_text()
    tree = ast.parse(text)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_resolve_gdn_backend_selection")
    seg = ast.get_source_segment(text, fn)
    ns: dict = {}
    # stubs que la funcion necesita
    envs = types.SimpleNamespace(VLLM_GDN_DECODE_KERNEL="triton")
    ns["os"] = os
    ns["envs"] = envs
    ns["is_flash_next_text_model_type"] = lambda mt: mt == "qwen3_8_flash_next_text"
    exec(compile(seg, "<selector>", "exec"), ns)
    return ns["_resolve_gdn_backend_selection"]


class Cfg:
    def __init__(self, additional_config, model_type="qwen3_8_flash_next_text"):
        self.additional_config = additional_config
        mc = types.SimpleNamespace()
        mc.hf_text_config = types.SimpleNamespace(model_type=model_type)
        self.model_config = mc


def case(sel, label, additional, mix_env, expect=None, raises=False):
    old = os.environ.get("QWEN_GDN_MIX")
    if mix_env is None:
        os.environ.pop("QWEN_GDN_MIX", None)
    else:
        os.environ["QWEN_GDN_MIX"] = mix_env
    try:
        got = sel(Cfg(additional))
        if raises:
            fail(f"{label}: esperaba ValueError, dio {got}")
        elif got[:2] != expect:
            fail(f"{label}: esperaba {expect}, dio {got[:2]}")
        else:
            ok(f"{label}: {got[:2]}")
    except ValueError:
        if raises:
            ok(f"{label}: ValueError (como se esperaba)")
        else:
            fail(f"{label}: ValueError inesperado, esperaba {expect}")
    finally:
        if old is None:
            os.environ.pop("QWEN_GDN_MIX", None)
        else:
            os.environ["QWEN_GDN_MIX"] = old


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_dir():
        print(f"fork-src no encontrado: {src}", file=sys.stderr)
        return 1

    # Con el parche en modo A activo (la funcion lee QWEN_GDN_MIX en runtime).
    root = patch_copy(src, "A")
    sel = load_selector(root)
    mix_add = {"gdn_prefill_backend": "triton", "gdn_decode_kernel": "b12x"}
    case(sel, "A mix (triton+b12x, QWEN_GDN_MIX=A)", mix_add, "A",
         expect=("triton", "b12x"))
    case(sel, "sin mix (triton+b12x) -> politica original", mix_add, None,
         raises=True)
    case(sel, "B mix (triton+b12x, QWEN_GDN_MIX=B)", mix_add, "B",
         expect=("triton", "b12x"))
    case(sel, "auto+b12x -> los dos b12x", {"gdn_decode_kernel": "b12x"}, "A",
         expect=("b12x", "b12x"))
    case(sel, "E2 triton+triton intacto",
         {"gdn_prefill_backend": "triton", "gdn_decode_kernel": "triton"}, "A",
         expect=("triton", "triton"))
    case(sel, "flashinfer+triton intacto (sin b12x)",
         {"gdn_prefill_backend": "flashinfer", "gdn_decode_kernel": "triton"}, "A",
         expect=("flashinfer", "triton"))

    # Sin parchear: la misma entrada triton+b12x con QWEN_GDN_MIX=A debe seguir
    # fallando (la imagen original no tiene el codigo de relaxation).
    raw = Path(tempfile.mkdtemp(prefix="sel-raw-"))
    d = raw / REL
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / REL, d)
    sel_raw = load_selector(raw)
    case(sel_raw, "imagen SIN parchear (triton+b12x, mix=A) -> ValueError",
         mix_add, "A", raises=True)

    print()
    if FAILURES:
        print(f"RESULTADO: {len(FAILURES)} FALLO(S)")
        return 1
    print("RESULTADO: LOGICA DEL SELECTOR OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
