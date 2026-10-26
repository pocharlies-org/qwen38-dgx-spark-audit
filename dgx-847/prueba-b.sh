#!/usr/bin/env bash
# DGX-847 prueba B: la 2× con --max-num-batched-tokens 4096 (el nuestro) en vez de 8192, y el mismo benchy de 64K (c1, c8).
# El flag va al final de la línea de vllm serve (EXTRA_ARGS): argparse se queda con el último.
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; R=http://spark2:8000
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
kubectl create configmap -n llm ursu-recipe --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh --dry-run=client -o yaml | kubectl apply -f -
kubectl label cm -n llm ursu-recipe dgx-847=prueba --overwrite >/dev/null
. $D/ops/espera-2x.sh; reinicia_2x EXTRA_ARGS="${EXTRA:---max-num-batched-tokens ${MNBT:-4096}}" || exit 1
log "reinicio de la 2× con: ${EXTRA:---max-num-batched-tokens ${MNBT:-4096}}"; t0=$(date +%s)
espera_2x || exit 1
log "2× lista en $(( ($(date +%s) - t0) / 60 )) min"
kubectl logs -n llm deploy/ursu-2x-head | grep -E 'attention block size|max_num_batched_tokens|compilation:|init engine|Error|error' | tail -3 | cut -c1-200
mkdir -p $D/pruebas/${TAG:-b-mnbt${MNBT:-4096}}
/home/user/bin/benchy-venv/bin/llama-benchy --base-url $R/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next \
  --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid --pp 2048 --tg 128 --depth 65535 --concurrency 1 8 --runs 3 \
  --enable-prefix-caching --format csv --save-result $D/pruebas/${TAG:-b-mnbt${MNBT:-4096}}/medium.csv > $D/pruebas/${TAG:-b-mnbt${MNBT:-4096}}/run.log 2>&1
log "benchy exit=$?"
