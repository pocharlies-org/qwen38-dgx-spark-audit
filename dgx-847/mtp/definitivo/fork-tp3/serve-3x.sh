#!/bin/bash
# DGX-852: TP=3 en el FORK de ursuciprian (ghcr.io/ursuciprian/spark-vllm-b12x@sha256:8abb60ee…) con
# la variante E2 (estado SSM en BF16 + GDN triton) y el vocabulario del borrador en castellano.
#
# Copia de tp3/recipe/serve-3x.sh (el original de DGX-847 no se toca) con cuatro cambios:
#  1. la vista que produce `mtp/definitivo/universal/construir-universal.sh` (`--universal` sobre el
#     checkpoint ORIGINAL: PLE en NVFP4 re-troceada, geometría y expertos rellenados a múltiplos de
#     3, vocabulario intacto), más la copia MXFP8 de `construir-vistas.sh` — sus formas son las
#     GLOBALES padreadas y `_tp_shard` la corta por rango, así que sirve también en TP=1 y TP=2.
#     Con `MODEL_DIR=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3-fork` sigue valiendo la vista
#     vieja (PLE en FP8): el preflight acepta los dos formatos. Preflight de formas porque nuestra
#     vista e768 NO carga en su fork y eso se entera uno por el log cuarenta minutos después de
#     lanzar (partes PLE de 2.500.012 y 248.448 filas).
#  2. E2: `--mamba-ssm-cache-dtype bfloat16`, `--gdn-decode-kernel triton` y el `--additional-config`
#     que fija triton también en el prefill. Con el GDN b12x el estado SSM tiene que ser FP32
#     (prueba E de DGX-847: ni arranca en BF16), así que el BF16 obliga a cambiar las dos cosas.
#  3. el vocabulario del borrador en castellano, del ConfigMap llm/qwen38fn-mtp-draft-vocab.
#  4. nada más: anillo NCCL del triángulo DGX-817, RoCE one-shot apagado y bootstrap por la LAN,
#     como en la 3x de DGX-847.
set -uo pipefail

MODEL_DIR=${MODEL_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-universal}
MXFP8_DIR=${MXFP8_DIR:-/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3}
DRAFT_VOCAB=${DRAFT_VOCAB:-/opt/mtp-vocab-es/ids-es-K131072.txt.gz}
[ -f "$MODEL_DIR/config.json" ] || { echo "ABORTA: falta la vista del checkpoint ($MODEL_DIR). Construye: /home/dibanerz/dgx-852/construir-universal.sh" >&2; exit 6; }
[ -f "$MXFP8_DIR/model.safetensors.index.json" ] || { echo "ABORTA: falta la vista TP=3 de la copia MXFP8 ($MXFP8_DIR)." >&2; exit 6; }
[ -s "$DRAFT_VOCAB" ] || { echo "ABORTA: no hay vocabulario del borrador en $DRAFT_VOCAB (ConfigMap llm/qwen38fn-mtp-draft-vocab montado en /opt/mtp-vocab-es)." >&2; exit 6; }

# Preflight de la vista: es LA DEL FORK, no la nuestra. Barato (leer dos cabeceras) y aborta antes.
python3 - "$MODEL_DIR" "$MXFP8_DIR" <<'PY' || exit 6
import json, struct, sys

def head(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))

mdir, cdir = sys.argv[1:3]
tc = json.load(open(f"{mdir}/config.json"))["text_config"]
err = []
ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
if tc.get("vocab_size") != 248320:
    err.append(f"vocab_size {tc.get('vocab_size')} != 248320 (el fork deriva de ahí los multiplicadores del hash PLE)")
if "tp3_orig_vocab_size" in tc:
    err.append("tp3_orig_vocab_size presente: es nuestra vista, no la del fork")
if tc.get("make_ngram_vocab_size_divisible_by") != 384:
    err.append(f"divisor de la PLE {tc.get('make_ngram_vocab_size_divisible_by')} != 384 (padded % 3 == 0)")
# La PLE admite dos formatos, y son las DOS vistas que circulan por aquí: la UNIVERSAL
# (`--universal`, NVFP4 re-troceado: 80 bytes de peso + 10 de escala por fila, una versión para
# TP=1/2/3) y la de antes (`--fork-view`, descuantizada a FP8 con escala global BF16). Las dos
# traen las mismas 2.500.014 filas por parte y el mismo divisor 384.
dtype_ple = tc.get("ple_embedding_dtype")
if dtype_ple == "nvfp4":
    ple_ok = [(f"{ple}.shard_0.weight", "U8", [2500014, 80]),
              (f"{ple}.shard_0.weight_scale", "F8_E4M3", [2500014, 10]),
              (f"{ple}.weight_scale_2", "F32", [1])]
elif dtype_ple == "float8_e4m3fn":
    ple_ok = [(f"{ple}.shard_0.weight", "F8_E4M3", [2500014, 160]),
              (f"{ple}.weight_scale", "BF16", [1])]
else:
    err.append(f"PLE {dtype_ple!r}: ni nvfp4 (vista universal) ni float8_e4m3fn (vista fork TP=3)")
    ple_ok = []
if tc.get("moe_intermediate_size") != 768 or tc.get("shared_expert_intermediate_size") != 768:
    err.append(f"intermedios {tc.get('moe_intermediate_size')}/{tc.get('shared_expert_intermediate_size')} != 768 "
               "(el backend b12x rechaza EP y 640 no se parte en 3)")
wm = json.load(open(f"{mdir}/model.safetensors.index.json"))["weight_map"]
for name, dtype, want in tuple(ple_ok) + (
                          ("model.language_model.embed_tokens.weight", "BF16", [248320, 2560]),
                          ("lm_head.weight", "BF16", [248320, 2560])):
    got = head(f"{mdir}/{wm[name]}")[name]
    if [got["dtype"], got["shape"]] != [dtype, want]:
        err.append(f"{name}: {got['dtype']} {got['shape']} != {dtype} {want}")
cw = json.load(open(f"{cdir}/model.safetensors.index.json"))["weight_map"]
cf = cw["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
c = head(f"{cdir}/{cf}")["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
if c["shape"] != [15360, 2560] or c["dtype"] != "F8_E4M3":
    err.append(f"copia MXFP8 in_proj_qkv: {c['dtype']} {c['shape']} != F8_E4M3 [15360, 2560] "
               "(_tp_shard corta la copia contra la partición de la capa)")
if len(cw) > 1000:
    err.append(f"index de la copia con {len(cw)} tensores: es el del checkpoint entero, apunta a ficheros que no están")
if err:
    print("ABORTA (vista TP=3 del fork): " + "; ".join(err), file=sys.stderr)
    sys.exit(1)
print(f"VISTA ok: vocab 248320, PLE 128x2500014 en {dtype_ple} ({len(ple_ok)} tensores comprobados), "
      f"moe/shared 768, copia MXFP8 15360x2560")
PY

# Semillas de b12x: la de la imagen esta claveada en el namespace a tensor_parallel=2 Y a la ruta del modelo,
# asi que en TP=3 falla entera (autotune frio). Se copia igual: las capas que no cambian de forma siguen sirviendo.
S=/opt/b12x-seed/preparation; D="$XDG_CACHE_HOME/b12x/compile/preparation"
if [ -d "$S" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$D" && for f in "$S"/*.json; do [ -e "$D/$(basename "$f")" ] || cp "$f" "$D/"; done; fi
A=/opt/b12x-seed/torch_compile_cache
if [ -d "$A" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$XDG_CACHE_HOME/vllm/torch_compile_cache" && cp -rn "$A/." "$XDG_CACHE_HOME/vllm/torch_compile_cache/" || true; fi

# Anillo NCCL del triangulo (DGX-817): el grafo y NCCL_IB_HCA los genera la imagen nccl-ring con el host de
# ring.conf de ESTE rango. Corre ANTES del preflight, que mira la lista que NCCL va a usar.
if [ -n "${QWEN_NCCL_RING_HOST:-}" ]; then
  RING_OUT=$(python3 -I /opt/nccl-ring/nccl_ring_graph.py --host "$QWEN_NCCL_RING_HOST" --channels 2 --out /tmp/nccl-graph.xml) || {
    echo "PREFLIGHT ABORTA: el generador del grafo falla con host='$QWEN_NCCL_RING_HOST'" >&2; exit 7; }
  while IFS= read -r ln; do export "${ln%%=*}=${ln#*=}"; done <<< "$RING_OUT"
  echo "NCCL_RING host=$QWEN_NCCL_RING_HOST NCCL_IB_HCA=${NCCL_IB_HCA:?el generador no dio NCCL_IB_HCA} NCCL_GRAPH_FILE=${NCCL_GRAPH_FILE:?el generador no dio NCCL_GRAPH_FILE}"
  cat "$NCCL_GRAPH_FILE"
fi
IB_SYSFS="${IB_SYSFS:-/sys/class/infiniband}"
IFS=, read -r -a HCAS <<< "${NCCL_IB_HCA#=}"
PRE_GID="" PRE_MTU=""
MIN_HCAS=2
if [ -n "${QWEN_NCCL_RING_HOST:-}" ]; then MIN_HCAS=4; fi
if [ "${#HCAS[@]}" -lt "$MIN_HCAS" ]; then
  echo "PREFLIGHT ABORTA: NCCL_IB_HCA='${NCCL_IB_HCA:-}' trae ${#HCAS[@]} HCA y hacen falta ${MIN_HCAS}" >&2; exit 7
fi
for hca in "${HCAS[@]}"; do
  port="$IB_SYSFS/$hca/ports/1"
  est=$(cat "$port/state" 2>/dev/null || echo desconocido)
  case "$est" in
    *ACTIVE*) ;;
    *) echo "PREFLIGHT ABORTA: $hca puerto 1 no esta ACTIVE (state='$est'). Cable DAC?" >&2; exit 7 ;;
  esac
  gid=""
  for i in $(ls "$port/gid_attrs/types" 2>/dev/null | sort -n || true); do
    [ "$(cat "$port/gid_attrs/types/$i" 2>/dev/null || true)" = "RoCE v2" ] || continue
    g=$(cat "$port/gids/$i" 2>/dev/null || true)
    case "$g" in 0000:0000:0000:0000:0000:ffff:*) gid=$i; break ;; esac
  done
  mtu=$(cat "$IB_SYSFS/$hca"/device/net/*/mtu 2>/dev/null || true)
  case "$gid:$mtu" in
    :*|*:|*:*[!0-9]*) echo "PREFLIGHT ABORTA: $hca sin GID RoCE v2 IPv4 o sin MTU (gid='$gid' mtu='$mtu')" >&2; exit 7 ;;
  esac
  PRE_GID="${PRE_GID:-$gid}" PRE_MTU="${PRE_MTU:-$mtu}"
  if [ "$gid" != "$PRE_GID" ] || [ "$mtu" != "$PRE_MTU" ]; then
    echo "PREFLIGHT ABORTA: $hca gid=$gid mtu=$mtu frente a gid=$PRE_GID mtu=$PRE_MTU del primero" >&2; exit 7
  fi
done
echo "PREFLIGHT ok rank=${NODE_RANK} hcas=${NCCL_IB_HCA#=} gid=${PRE_GID} mtu=${PRE_MTU}"

# Su all-reduce RoCE en el anillo sin switch (roce-anillo/): parche de arranque del proxy + mapa por par. Sin
# B12X_ROCE_RING=1 no se toca nada y sigue NCCL (env-3x.env lo trae a 0).
if [ "${B12X_ROCE_RING:-0}" = 1 ]; then
  python3 -I /etc/ursu/patch_roce_ring.py || exit 3
  set -a; eval "$(python3 -I /etc/ursu/roce_ring_map.py --host "$QWEN_NCCL_RING_HOST")" || exit 3; set +a
  export B12X_ROCE_GID_INDEX=$PRE_GID VLLM_ENABLE_ROCE_ALLREDUCE=1 VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB
  echo "ROCE anillo rank=${NODE_RANK} hca=${B12X_ROCE_HCA} pares=${B12X_ROCE_PEER_HCAS} gid=${B12X_ROCE_GID_INDEX}"
fi

# E2: estado SSM en BF16 con el GDN triton por los dos lados. En BF16 el b12x no vale (exige FP32) y
# los checkpoints GDN diferidos tampoco (resolve() solo los admite con b12x en decode y prefill), por
# eso VLLM_GDN_DEFERRED_CHECKPOINTS=0 en env-3x.env.
exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --tensor-parallel-size 3 \
  --pipeline-parallel-size 1 \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype bfloat16 \
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
  --gdn-decode-kernel triton \
  --additional-config '{"gdn_prefill_backend":"triton","gdn_decode_kernel":"triton"}' \
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
  --nnodes 3 --node-rank $NODE_RANK --master-addr spark2 --master-port 25000 $HEADLESS ${EXTRA_ARGS:-}
