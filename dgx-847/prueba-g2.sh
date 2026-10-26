#!/usr/bin/env bash
# DGX-847 prueba G2 (pedida por Dani tras G): la 2× con la PLE en disco y --gpu-memory-utilization 0.7 en vez de 0.8, para que la
# KV no se coma la caché de páginas donde vive la tabla PLE (~12 GB por nodo). Si la pérdida de G desaparece, la causa era esa.
# Después: fuera la 2× y vuelta del residente (el PostSync llm-deploy-resume no corrió en la sync de las 07:46Z: se lanza a mano).
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; R=http://spark2:8000; T=$D/pruebas/g2-ple-disco-gmu07; mkdir -p $T
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
. $O/espera-2x.sh
kubectl create configmap -n llm ursu-recipe --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap -n llm ursu-env-2x --from-env-file=$O/recipe/env-2x.env --from-env-file=$O/recipe/env-sparkrun-2x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label cm -n llm ursu-recipe ursu-env-2x dgx-847=prueba --overwrite >/dev/null
# solo los dos Deployments de la 2×, a 0 réplicas (reinicia_2x los sube juntos)
python3 -c "
import sys; docs=open('$O/ursu.yaml').read().split('\n---')
print('\n---'.join(d.replace('replicas: 1','replicas: 0') for d in docs if 'name: ursu-2x-' in d))" | kubectl apply -f -
reinicia_2x VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_WILLNEED_MAX=8192 VLLM_PLE_MMAP_KEEPALIVE_MS=50 VLLM_PLE_MMAP_PREFILL_WILLNEED=1 EXTRA_ARGS="--gpu-memory-utilization 0.7" || exit 1
log "G2: 2× arriba con la PLE en disco y gpu-memory-utilization 0.7"
espera_2x || exit 1
kubectl logs -n llm deploy/ursu-2x-head > $T/arranque.log 2>&1
grep -E 'GPU KV cache size|Available KV cache|gpu_memory_utilization|gpu-memory-utilization' $T/arranque.log | cut -c1-200 | tail -3
for h in dgx dgx2; do log "$h: $(ssh $h 'free -g | sed -n 2p')"; done
TOOLS=/home/user/bench-results/dgx-768/tools
python3 $TOOLS/qwen38_decode_sweep.py --base-url $R --model qwen3.8-flash-next --concurrency 1,8 --warmup 30 --duration 120 --reps 2 \
  --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
  --max-model-len 262144 --seed 496368 --out $T/decode-agg.json > $T/decode-agg.log 2>&1
log "decode exit=$?"; tr '\r' '\n' < $T/decode-agg.log | grep -E '^C= ' | cut -c1-120
/home/user/bin/benchy-venv/bin/llama-benchy --base-url $R/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next \
  --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 8 --runs 3 \
  --enable-prefix-caching --format csv --save-result $T/medium.csv > $T/benchy.log 2>&1
log "benchy exit=$?"
for h in dgx dgx2; do log "$h: $(ssh $h 'free -g | sed -n 2p')"; done
log "G2: fin"
