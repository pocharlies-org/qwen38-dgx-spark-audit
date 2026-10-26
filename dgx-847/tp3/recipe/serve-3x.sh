#!/bin/bash
# DGX-847: la receta de ursuciprian en TP=3 (dgx2 head + dgx1 + dgx3). Es recipe/serve-2x.sh con cuatro cambios:
# la vista TP=3 del checkpoint (CONVERTIR.md), --nnodes 3 con arranque por la LAN (10.0.0.2 no llega a dgx3:
# cada enlace del triangulo es su propia /30), el anillo NCCL de DGX-817 (libnccl parcheada + grafo por rango)
# y el RoCE one-shot apagado (su proxy monta la malla por INDICE de HCA y el triangulo no la da).
# Flags, cuantizado, MTP=4 y tamaños de lote son los de la receta 2x, tal cual.
set -uo pipefail

MODEL_DIR=${MODEL_DIR:-/cache/huggingface/tp3/ursu-qwen38fn-tp3}
MXFP8_DIR=${MXFP8_DIR:-/cache/huggingface/tp3/ursu-mxfp8-tp3}
[ -f "$MODEL_DIR/config.json" ] || { echo "ABORTA: falta la vista TP=3 del checkpoint principal ($MODEL_DIR). Ver CONVERTIR.md." >&2; exit 6; }
[ -f "$MXFP8_DIR/model.safetensors.index.json" ] || { echo "ABORTA: falta la vista TP=3 de la copia MXFP8 ($MXFP8_DIR)." >&2; exit 6; }

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

exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --tensor-parallel-size 3 \
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
  --nnodes 3 --node-rank $NODE_RANK --master-addr spark2 --master-port 25000 $HEADLESS
