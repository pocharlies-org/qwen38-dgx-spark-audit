#!/usr/bin/env python3
"""Instala el dial de refusal (ablacion rank-1) sobre el FORK b12x de ursuciprian.

Mismo procedimiento que el mod de produccion (`k8s-ai-pocharlies:
k8s/qwen38-flash-next-ursucipian-mod/install.py`) sobre la nightly de RadixArk,
adaptado a tres diferencias reales:

 1. NO hay overlays. Los 5 overlays del mod de produccion (qsa.py, ops_qsa.py,
    qsa_cache.py, ple.py, platforms/interface.py, modelopt.py) son fixes nuestros
    de la nightly; el fork los tiene o no los usa (trae su propio qsa.py de 576 B
    y sus kernels b12x). Aqui solo se parchea, no se sustituye ningun fichero.
 2. El modulo del modelo se llama `models/qwen3_8_flash_next/`, no
    `models/qwen4_exp/nvidia/`. Todas las anclas van re-ubicadas (ver
    patch_vllm_qwen38next.py).
 3. El arbol llega con AOT compilado en caliente (`VLLM_USE_AOT_COMPILE=1` +
    `VLLM_USE_MEGA_AOT_ARTIFACT=1` + cache sembrada desde /opt/b12x-seed). El
    parche cambia `co_firstlineno` de las funciones compiladas, asi que vLLM no
    encuentra el artefacto y recompila: el PRIMER arranque parcheado paga la
    compilacion (los arranques calientes del window sin parche tardan 319 s en
    2x y 208 s en 1x). Con `VLLM_FORCE_AOT_LOAD=1` eso seria un error duro en
    vez de una recompila, y por eso se aborta aqui.

Idempotente: si `/opt/refusal/b12x-installed.json` existe, se limitan a
verificarse los hashes post-parche y se sale en 0.
"""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

site = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
# REFUSAL_ROOT existe solo para poder correr este instalador sin root contra un
# arbol de prueba en seco (CPU). En el pod no se pone: /opt/refusal.
root = Path(os.environ.get("REFUSAL_ROOT", "/opt/refusal"))
marker = root / "b12x-installed.json"
if marker.exists():
    for name, digest in json.loads(marker.read_text()).items():
        assert hashlib.sha256((site / name).read_bytes()).hexdigest() == digest, name
    print("b12x refusal mod already applied; hashes verified")
    sys.exit(0)

if os.environ.get("VLLM_FORCE_AOT_LOAD") == "1":
    raise RuntimeError(
        "VLLM_FORCE_AOT_LOAD=1: el parche invalida el artefacto AOT y con esto "
        "vLLM aborta en vez de recompilar. Dejalo en 0 (default del fork)."
    )

bundle = Path(__file__).parent
for name, digest in json.loads((bundle / "base-hashes.json").read_text()).items():
    if name.startswith("_"):
        continue
    if hashlib.sha256((site / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(
            f"Fuente del fork distinta de la verificada: {name} "
            "(re-genera base-hashes.json contra la imagen real)"
        )

shutil.copytree(bundle / "refusal", root, dirs_exist_ok=True)
# Se instala como root pero se sirve como usuario sin privilegios.
root.chmod(0o755)
for path in root.rglob("*"):
    path.chmod(0o755 if path.is_dir() else 0o644)
(root / "SITE").write_text(str(site))

router = site / "entrypoints/serve/refusal"
router.mkdir(exist_ok=True)
for source, target in [("refusal_init.py", "__init__.py"),
                       ("refusal_api_router.py", "api_router.py")]:
    shutil.copyfile(root / "payload" / source, router / target)

# patch_vllm_probe.py el ULTIMO: dos de sus anclas caen en ficheros que los
# otros dos ya parchearon.
for script, extra in [
    ("patch_vllm_qwen38next.py", ["--direction", str(root / "refusal_direction_flashnext.npy")]),
    ("patch_vllm_runtime.py", []),
    ("patch_vllm_probe.py", []),
]:
    subprocess.run(
        [sys.executable, str(root / script), "--site", str(site),
         "--payload", str(root / "payload"), *extra],
        check=True,
    )

env = dict(os.environ,
           VLLM_REFUSAL_DIRECTION=str(root / "refusal_direction_flashnext.npy"),
           VLLM_REFUSAL_LAMBDA="0")
env.pop("PYTHONPATH", None)
with tempfile.TemporaryDirectory(prefix="refusal-cpu-tests-") as cache:
    for variable in ["FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                     "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME"]:
        env[variable] = str(Path(cache) / variable.lower())
    skip = os.environ.get("B12X_SKIP_CPU_TESTS") == "1"
    for name in ([] if skip else [
        "test_projection_equivalence.py",
        "test_per_request_lambda.py",
        "test_cache_salt_contract.py",
        "test_hc_shape_contract.py",
        "test_lambda_zero_identity.py",
        "check_injected_scope.py",
        "check_guard_counts.py",
        "check_lambda_survives.py",
        "test_refusal_probe.py",
    ]):
        args = [str(site)] if name.startswith("check_") else []
        subprocess.run([sys.executable, str(root / name), *args],
                       cwd=root, env=env, check=True)

paths = [
    "models/qwen3_8_flash_next/model.py",
    "models/qwen3_8_flash_next/mtp.py",
    "v1/core/sched/output.py",
    "v1/core/kv_cache_utils.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/refusal_utils.py",
    "v1/worker/gpu_worker.py",
    "entrypoints/serve/__init__.py",
    "entrypoints/serve/refusal/__init__.py",
    "entrypoints/serve/refusal/api_router.py",
    "refusal_projection.py",
    "refusal_probe.py",
    "v1/worker/gpu/async_utils.py",
    "v1/outputs.py",
    "v1/core/sched/scheduler.py",
    "v1/engine/__init__.py",
    "v1/engine/output_processor.py",
    "outputs.py",
    "entrypoints/openai/chat_completion/serving.py",
]
hashes = {}
for name in dict.fromkeys(paths):
    p = site / name
    ast.parse(p.read_text())
    hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
marker.write_text(json.dumps(hashes, indent=2) + "\n")
print("b12x refusal mod applied and CPU contracts passed")
