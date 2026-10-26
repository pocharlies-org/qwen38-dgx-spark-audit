#!/usr/bin/env bash
# DGX-847 prueba G: la 2× con la PLE fuera de la GPU (su VLLM_PLE_MMAP: cada nodo lee su mitad de la tabla del NVMe por la caché
# de páginas, el mismo mecanismo que nuestro DGX-773) y su receta por lo demás. Mide si se pierde rendimiento: decode
# c1/4/8/10 (mismos argumentos que measure.py) y benchy de 64K; y cuánto KV gana con la memoria liberada.
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; R=http://spark2:8000; T=$D/pruebas/g-ple-disco; mkdir -p $T
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
trap 'log "G: fin"' EXIT
until grep -q 'f-nuestros-3: fin' $D/pruebas/ef.log 2>/dev/null; do sleep 30; done
. $D/ops/espera-2x.sh; reinicia_2x EXTRA_ARGS- VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_WILLNEED_MAX=8192 VLLM_PLE_MMAP_KEEPALIVE_MS=50 VLLM_PLE_MMAP_PREFILL_WILLNEED=1 || exit 1
log "reinicio de la 2× con la PLE en disco"; t0=$(date +%s)
espera_2x || exit 1
log "2× lista en $(( ($(date +%s) - t0) / 60 )) min"
kubectl logs -n llm deploy/ursu-2x-head > $T/arranque.log 2>&1
grep -E 'ple_mmap|PLE page-cache|attention block size|GPU KV cache size|Available KV cache|init engine' $T/arranque.log | cut -c1-200
for h in dgx dgx2; do log "$h: $(ssh $h 'free -g | sed -n 2p')"; done
TOOLS=/home/user/bench-results/dgx-768/tools
python3 $TOOLS/qwen38_decode_sweep.py --base-url $R --model qwen3.8-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
  --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
  --max-model-len 262144 --seed 496368 --out $T/decode-agg.json > $T/decode-agg.log 2>&1
log "decode exit=$?"; tr '\r' '\n' < $T/decode-agg.log | grep -E '^C= ' | cut -c1-120
/home/user/bin/benchy-venv/bin/llama-benchy --base-url $R/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next \
  --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 8 --runs 3 \
  --enable-prefix-caching --format csv --save-result $T/medium.csv > $T/benchy.log 2>&1
log "benchy exit=$?"
for h in dgx dgx2; do log "$h: $(ssh $h 'free -g | sed -n 2p')"; done
