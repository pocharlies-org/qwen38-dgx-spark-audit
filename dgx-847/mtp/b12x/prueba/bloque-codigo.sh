B12X_WHEELHOUSE="${B12X_WHEELHOUSE:-/cache/huggingface/wheelhouse/b12x-1.3.0}"
B12X_TP=$(printf '%s\n' "${ARGS[@]}" | awk 'take { print; take = 0 } $0 == "--tensor-parallel-size" { take = 1 }')
if [ "${QWEN38_B12X:-0}" = "1" ]; then
  B12X_WHEELHOUSE="$B12X_WHEELHOUSE" B12X_TP="$B12X_TP" B12X_MOE="${QWEN38_B12X_MOE:-0}" \
  MODEL_PATH="$MODEL_PATH" \
    python3 - <<'PYEOF' || { echo "QWEN38_B12X ABORTA (73): ver el mensaje de arriba" >&2; exit 73; }
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import zipfile

WHEELHOUSE = pathlib.Path(os.environ["B12X_WHEELHOUSE"])
WHEEL = "b12x-1.3.0-py3-none-any.whl"
# Los seis submodulos que vllm.utils.b12x importa al cargar (utils/b12x.py:33-42). Si uno no
# importa, vLLM lo trata como "b12x no disponible" y el backend pedido falla tarde y mal.
SUBMODULOS = (
    "b12x.attention.paged",
    "b12x.gemm.blockscaled",
    "b12x._lib.intrinsics",
    "b12x.gemm.mxfp8_linear",
    "b12x.gemm.tensor_fp8_linear",
    "b12x.moe.fused_moe",
)


def abort(motivo):
    print(f"QWEN38_B12X ABORTA: {motivo}", file=sys.stderr)
    sys.exit(1)


# 1. La rueda y su sha256 contra SHA256SUMS si el wheelhouse lo trae.
ruta = WHEELHOUSE / WHEEL
if not ruta.is_file():
    abort(f"no hay {WHEEL} en {WHEELHOUSE} (el NFS se monta de solo lectura: se copia en dgx1)")
sums = WHEELHOUSE / "SHA256SUMS"
real = hashlib.sha256(ruta.read_bytes()).hexdigest()
if sums.is_file():
    esperado = {p.split()[1]: p.split()[0] for p in sums.read_text().splitlines() if p.strip()}
    if esperado.get(WHEEL) not in (None, real):
        abort(f"sha256 de la rueda no cuadra: {real} != {esperado[WHEEL]}")
print(f"QWEN38_B12X rueda ok sha256={real[:16]}…")

# 2. Lo que DECLARA la rueda (METADATA dentro del .whl, no una lista nuestra copiada aqui),
#    contra lo instalado. Con --no-deps pip no baja nada, asi que si algo faltara hay que
#    decirlo aqui y no descubrirlo en el log del arranque.
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    abort("la imagen no trae packaging (hace falta para leer los Requires-Dist de la rueda)")
with zipfile.ZipFile(ruta) as z:
    meta = z.read("b12x-1.3.0.dist-info/METADATA").decode()
faltan, cumplidas = [], []
for linea in (l for l in meta.splitlines() if l.startswith("Requires-Dist:")):
    req = Requirement(linea.split(":", 1)[1].strip())
    if req.marker and not req.marker.evaluate({"extra": ""}):
        continue  # extras de desarrollo (pytest), no cuentan
    try:
        instalado = Version(md.version(req.name))
    except md.PackageNotFoundError:
        faltan.append(f"{req} (AUSENTE)")
        continue
    if req.specifier and not req.specifier.contains(instalado, prereleases=True):
        faltan.append(f"{req} -> instalado {instalado}")
    else:
        cumplidas.append(f"{req.name}={instalado}")
if faltan:
    abort("b12x pide y la imagen no cumple: " + "; ".join(faltan))
print(f"QWEN38_B12X dependencias ok ({len(cumplidas)}): " + ", ".join(cumplidas))

# 3. Instalar, o admitir que ya esta (el pod puede reiniciar dentro de la ventana).
if importlib.util.find_spec("b12x") is None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--disable-pip-version-check", "--quiet", "--find-links", str(WHEELHOUSE),
             "b12x==1.3.0"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        abort(f"pip install salio con {e.returncode}")
    print(f"QWEN38_B12X instalado b12x=={md.version('b12x')} desde {WHEELHOUSE}")
elif md.version("b12x") != "1.3.0":
    abort(f"b12x ya instalado, pero en otra version: {md.version('b12x')}")
else:
    print("QWEN38_B12X b12x==1.3.0 ya estaba instalado en este contenedor")

# 4. Que los seis submodulos que va a tocar vLLM importen.
for nombre in SUBMODULOS:
    try:
        importlib.import_module(nombre)
    except Exception as e:
        abort(f"import {nombre} falla: {type(e).__name__}: {e}")
print(f"QWEN38_B12X imports ok ({len(SUBMODULOS)} submodulos)")

# 5. El gating de b12x en ESTA GPU y, si se pide, la geometria del checkpoint para el MoE.
import torch
from b12x._lib.gating import is_b12x

if not torch.cuda.is_available():
    abort("sin CUDA visible: no se puede comprobar el gating de b12x")
if not is_b12x():
    abort(f"b12x solo corre en SM120/SM121 y esto es {torch.cuda.get_device_capability()}")
if not importlib.import_module("b12x.gemm.blockscaled").is_supported():
    abort("b12x.gemm.blockscaled.is_supported() es False en esta maquina")
print(f"QWEN38_B12X gating ok cap={torch.cuda.get_device_capability()} "
      f"cutlass-dsl={md.version('nvidia-cutlass-dsl')} triton={md.version('triton')}")

if os.environ.get("B12X_MOE") == "1":
    if not importlib.import_module("b12x.moe.fused_moe").is_supported():
        abort("b12x.moe.fused_moe.is_supported() es False en esta maquina")
    cfg = json.loads((pathlib.Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    moe_i, tp = int(tc["moe_intermediate_size"]), int(os.environ["B12X_TP"])
    if moe_i % tp:
        abort(
            f"QWEN38_B12X_MOE quita EP y entonces el MoE se parte por TP, pero "
            f"moe_intermediate_size={moe_i} no es divisible por tp={tp} "
            f"(fused_moe/config.py:1346). Hace falta una vista con los expertos enrutados "
            f"rellenos a {((moe_i + tp - 1) // tp) * tp}; esta es {os.environ['MODEL_PATH']}"
        )
    print(f"QWEN38_B12X MoE ok moe_intermediate_size={moe_i} % tp={tp} == 0")
PYEOF
  ARGS+=( --linear-backend b12x )
  if [ "${QWEN38_B12X_MOE:-0}" = "1" ]; then
    # b12x rechaza EP: fuera --enable-expert-parallel y su --all2all-backend del array.
    SIN_EP=()
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
      case "${ARGS[i]}" in
        --enable-expert-parallel) ;;
        --all2all-backend) i=$((i + 1)) ;;
        *) SIN_EP+=("${ARGS[i]}") ;;
      esac
    done
    ARGS=("${SIN_EP[@]}")
    ARGS+=( --moe-backend b12x )
  fi
  export B12X_COMPILE_CACHE_DIR="${B12X_COMPILE_CACHE_DIR:-/tmp/b12x-compile}"
  mkdir -p "$B12X_COMPILE_CACHE_DIR"
  echo "QWEN38_B12X activo rank=${VLLM_NODE_RANK} linear=b12x moe=${QWEN38_B12X_MOE:-0} wheelhouse=${B12X_WHEELHOUSE} jit=${B12X_COMPILE_CACHE_DIR}"
fi
