#!/bin/bash
# DGX-852: variante de serve-2x.sh que sirve la VISTA UNIVERSAL (`--universal`, PLE en NVFP4) por
# RUTA, en vez del id de HF con su revisión. Es la misma receta de siempre con tres cambios:
#  1. `vllm serve "$MODEL_DIR"` (sin `--revision`): una sola versión para TP=1, 2 y 3, la que sale
#     de `mtp/definitivo/universal/construir-universal.sh` en dgx1 y que se ve por el PVC ro
#     `hf-cache-nvidia-dgx-ro` montado en /cache/huggingface.
#  2. la copia MXFP8 pasa a ser la PADREADA (`qwen38fn-mtp-ursu-mxfp8-tp3`), no el shard 35 suelto
#     del hub: `_tp_shard` corta la copia contra la partición de la capa y las formas tienen que
#     casar. Se exporta aquí mismo porque `env-2x.env` sigue apuntando al hub (la receta vieja).
#  3. preflight de la vista por forma y divisibilidad: aborta en vez de enterarse a los 40 minutos.
# Variables de entorno nuevas: MODEL_DIR (por defecto /cache/huggingface/qwen38fn-mtp-ursu-universal)
# y MXFP8_DIR (/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3). Lo demás, idéntico a serve-2x.sh.
set -uo pipefail
MODEL_DIR=${MODEL_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-universal}
MXFP8_DIR=${MXFP8_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3}
[ -f "$MODEL_DIR/config.json" ] || { echo "ABORTA: falta la vista universal en $MODEL_DIR. Construye en dgx1: /home/dibanerz/dgx-852/construir-universal.sh" >&2; exit 6; }
[ -f "$MXFP8_DIR/model.safetensors.index.json" ] || { echo "ABORTA: falta la copia MXFP8 padreada en $MXFP8_DIR" >&2; exit 6; }
python3 - "$MODEL_DIR" "$MXFP8_DIR" 2 <<'PY' || exit 6
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
PY
export VLLM_B12X_NVFP4_MXFP8_CHECKPOINT="$MXFP8_DIR"
S=/opt/b12x-seed/preparation; D="$XDG_CACHE_HOME/b12x/compile/preparation"; if [ -d "$S" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$D" && for f in "$S"/*.json; do [ -e "$D/$(basename "$f")" ] || cp "$f" "$D/"; done; fi
A=/opt/b12x-seed/torch_compile_cache; if [ -d "$A" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$XDG_CACHE_HOME/vllm/torch_compile_cache" && cp -rn "$A/." "$XDG_CACHE_HOME/vllm/torch_compile_cache/" || true; fi
exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --tensor-parallel-size 2 \
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
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
  --gpu-memory-utilization 0.8 \
  --nnodes 2 --node-rank $NODE_RANK --master-addr 10.0.0.2 --master-port 25000 $HEADLESS ${EXTRA_ARGS:-}
