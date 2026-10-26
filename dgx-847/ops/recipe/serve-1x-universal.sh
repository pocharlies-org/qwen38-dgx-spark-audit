#!/bin/bash
# DGX-852: variante de serve-1x.sh que sirve la VISTA UNIVERSAL (`--universal`, PLE en NVFP4) por
# RUTA, en vez del id de HF con su revisión (03f4a057). Mismos tres cambios que serve-2x-universal.sh:
#  1. `vllm serve "$MODEL_DIR"` sin `--revision`.
#  2. la copia MXFP8 es la padreada (`qwen38fn-mtp-ursu-mxfp8-tp3`), exportada aquí porque
#     `env-1x.env` sigue apuntando al hub.
#  3. preflight de formas y divisibilidad por el tp de esta receta.
# Variables de entorno nuevas: MODEL_DIR (/cache/huggingface/qwen38fn-mtp-ursu-universal) y
# MXFP8_DIR (/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3).
#
# OJO TP=1 (lo mide `check --universal` sobre el checkpoint real): la vista universal son 111,3 GiB
# de pesos frente a 97,7 GiB del ORIGINAL, porque los expertos van rellenados 640->768 para que
# partan en 3 y ese relleno en TP=1 sobra (+13,6 GiB). Un Spark con 121 GiB ya anda a 6 GiB libres
# con el ORIGINAL y los 14 GiB de KV de esta receta (--kv-cache-memory-bytes 15032385536): la vista
# universal NO cabe sin bajar el KV por debajo de ~1 GiB. El preflight lo avisa; la decisión es de
# quien lanza (bajar el KV o seguir sirviendo el ORIGINAL en TP=1, que es la MISMA versión NVFP4
# sin el relleno de TP=3).
set -uo pipefail
MODEL_DIR=${MODEL_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-universal}
MXFP8_DIR=${MXFP8_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3}
[ -f "$MODEL_DIR/config.json" ] || { echo "ABORTA: falta la vista universal en $MODEL_DIR. Construye en dgx1: /home/dibanerz/dgx-852/construir-universal.sh" >&2; exit 6; }
[ -f "$MXFP8_DIR/model.safetensors.index.json" ] || { echo "ABORTA: falta la copia MXFP8 padreada en $MXFP8_DIR" >&2; exit 6; }
python3 - "$MODEL_DIR" "$MXFP8_DIR" 1 <<'PY' || exit 6
import json, os, struct, sys

def head(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return {k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"}

mdir, cdir, tp = sys.argv[1], sys.argv[2], int(sys.argv[3])
tc = json.load(open(f"{mdir}/config.json"))["text_config"]
ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
wm = json.load(open(f"{mdir}/model.safetensors.index.json"))["weight_map"]
need = [f"{ple}.shard_0.weight", f"{ple}.shard_0.weight_scale", f"{ple}.weight_scale_2"]
cw = json.load(open(f"{cdir}/model.safetensors.index.json"))["weight_map"]
cq = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
# antes de leer nada: si faltan los tensores o la copia es el index entero (apunta a ficheros que
# no están), el abort tiene que decir por qué, no soltar un traceback
if faltan := [t for t in need if t not in wm]:
    print(f"ABORTA (TP={tp}): la vista no trae {faltan}; son los nombres de la UNIVERSAL en NVFP4 "
          "(la vista FP8 renombra weight_scale_2 a weight_scale y no trae escalas por shard)", file=sys.stderr)
    sys.exit(1)
if cq not in cw or not os.path.exists(f"{cdir}/{cw[cq]}") or len(cw) > 1000:
    print(f"ABORTA (TP={tp}): la copia MXFP8 {cdir} no es la vista padreada ({len(cw)} tensores en "
          "su index; la buena son ~480 y un solo fichero)", file=sys.stderr)
    sys.exit(1)
hw, hs, hg = (head(f"{mdir}/{wm[t]}") for t in need)
def par(h, t):
    return [h[t]["dtype"], h[t]["shape"]]
checks = [
    ("dtype de la PLE", tc.get("ple_embedding_dtype"), "nvfp4"),
    ("vocabulario", tc.get("vocab_size"), 248320),
    ("tp3_orig_vocab_size fuera", "tp3_orig_vocab_size" in tc, False),
    ("divisor de la PLE", tc.get("make_ngram_vocab_size_divisible_by"), 384),
    (f"{ple}.shard_0.weight", par(hw, f"{ple}.shard_0.weight"), ["U8", [2500014, 80]]),
    (f"{ple}.shard_0.weight_scale", par(hs, f"{ple}.shard_0.weight_scale"), ["F8_E4M3", [2500014, 10]]),
    (f"{ple}.weight_scale_2", par(hg, f"{ple}.weight_scale_2"), ["F32", [1]]),
    ("copia MXFP8 in_proj_qkv", par(head(f"{cdir}/{cw[cq]}"), cq), ["F8_E4M3", [15360, 2560]]),
]
mal = [f"{k}: {v} != {w}" for k, v, w in checks if v != w]
for k in ("num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads",
          "shared_expert_intermediate_size", "moe_intermediate_size"):
    if tc.get(k, 0) % tp:
        mal.append(f"TP={tp}: {k}={tc.get(k)} no divide")
padded = hw[f"{ple}.shard_0.weight"]["shape"][0] * int(tc["split_ngram_parts"])
if padded % tp:
    mal.append(f"TP={tp}: padded_vocab_size {padded} no divide (_contracts.py::storage_layout)")
if mal:
    print(f"ABORTA (vista universal en TP={tp}): " + "; ".join(mal), file=sys.stderr)
    sys.exit(1)
print(f"VISTA ok TP={tp}: PLE nvfp4 {tc['split_ngram_parts']}x{hw[f'{ple}.shard_0.weight']['shape'][0]} "
      f"(80+10 B/fila), tabla {padded:,} divide entre {tp}, vocab 248320, divisor 384, moe/shared 768")
if tp == 1:
    print("AVISO TP=1: 111,3 GiB de pesos (los 768 de los expertos sobran en un solo rango); el "
          "ORIGINAL son 97,7 GiB y esta receta pide 14 GiB de KV sobre 121 GiB del Spark.")
PY
export VLLM_B12X_NVFP4_MXFP8_CHECKPOINT="$MXFP8_DIR"
S=/opt/b12x-seed/preparation; D="$XDG_CACHE_HOME/b12x/compile/preparation"; if [ -d "$S" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$D" && for f in "$S"/*.json; do [ -e "$D/$(basename "$f")" ] || cp "$f" "$D/"; done; fi
A=/opt/b12x-seed/torch_compile_cache; if [ -d "$A" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$XDG_CACHE_HOME/vllm/torch_compile_cache" && cp -rn "$A/." "$XDG_CACHE_HOME/vllm/torch_compile_cache/" || true; fi
exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-parallel-prefills 4 \
  --prefill-policy decode-aware \
  --decode-refill-target auto \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format b12x \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
  --kv-cache-memory-bytes 15032385536 \
  --gpu-memory-utilization 0.01 ${EXTRA_ARGS:-}
