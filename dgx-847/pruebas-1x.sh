#!/usr/bin/env bash
# DGX-847: las pruebas de contexto largo y PLE en su receta de 1 Spark (dgx3), en paralelo con las de la 2×.
# A (caché sobre 64K) → B (4096 por paso) → F (4096 + 2048 por petición) → E (SSM en BF16; se espera el mismo fallo que en la
# 2×) → G (la PLE en GPU: su 1× ya la lleva en disco, así se ve lo que cuesta el disco) → C (perfil). Un fallo de arranque en un
# solo nodo es de la configuración: se registra y se pasa a la siguiente. Al salir escribe «1x: fin» (C lo espera para cerrar).
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; R=http://spark3:8000; T=$D/pruebas/1x; mkdir -p $T
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
trap 'log "1x: fin"' EXIT
BENCHY=/home/user/bin/benchy-venv/bin/llama-benchy
benchy() {  # $1 = directorio, $2.. = profundidades
  local o=$1; shift
  $BENCHY --base-url $R/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth "$@" --concurrency 1 8 --runs 3 --enable-prefix-caching --format csv --save-result $o/medium.csv > $o/run.log 2>&1
  log "benchy exit=$?"
}
reinicia_1x() {  # baja, espera la memoria de dgx3 (NVRM la devuelve tarde), aplica el env y sube; espera /health (tope 40 min)
  kubectl scale -n llm deploy/ursu-1x --replicas=0 >/dev/null
  while kubectl get pods -n llm -l app=ursu-1x --no-headers 2>/dev/null | grep -q .; do sleep 5; done
  local t0=$(date +%s)
  until [ "$(ssh dgx3 "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")" -ge 100 ]; do
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "dgx3 no recupera la memoria"; return 1; }; sleep 10; done
  kubectl set env -n llm deploy/ursu-1x "$@" >/dev/null
  kubectl scale -n llm deploy/ursu-1x --replicas=1 >/dev/null
  log "1× arriba con: $*"; t0=$(date +%s); sleep 20
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R/health)" = 200 ]; do
    if kubectl get pods -n llm -l app=ursu-1x -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | grep -qE '[1-9]'; then
      log "no arranca: $(kubectl logs -n llm deploy/ursu-1x --previous 2>/dev/null | grep -E 'Error:|ValueError|RuntimeError' | grep -v 'APIServer pid=1)     ' | head -2 | cut -c1-250)"; return 1; fi
    [ $(( $(date +%s) - t0 )) -gt 2400 ] && { log "sin /health en 40 min"; return 1; }; sleep 15
  done
  log "1× lista en $(( ($(date +%s) - t0) / 60 )) min; $(kubectl logs -n llm deploy/ursu-1x | grep -oE 'attention block size to [0-9]+|GPU KV cache size: [0-9,]+ tokens' | sort -u | paste -sd' ')"
  return 0
}
kubectl create configmap -n llm ursu-recipe --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label cm -n llm ursu-recipe dgx-847=prueba --overwrite >/dev/null

log "A: caché sobre 64K (su receta tal cual)"
python3 -I $D/prueba-a-cache.py $R $T/a-cache-ctx65535.json 65535 | sed -u "s/^/A: /"
for t in "b-mnbt4096|--max-num-batched-tokens 4096" "f-nuestros-3|--max-num-batched-tokens 4096 --long-prefill-token-threshold 2048" "e-ssm-bf16|--mamba-ssm-cache-dtype bfloat16"; do
  TAG=${t%%|*}; mkdir -p $T/$TAG; log "$TAG: inicio"
  if reinicia_1x EXTRA_ARGS="${t#*|}"; then
    benchy $T/$TAG 65535
    python3 -I $D/prueba-a-cache.py $R $T/$TAG/a-cache-ctx65535.json 65535 | sed -u "s/^/$TAG A: /"
  else kubectl logs -n llm deploy/ursu-1x --previous --tail=60 > $T/$TAG/run.log 2>&1; fi
  log "$TAG: fin"
done
TAG=g-ple-gpu; mkdir -p $T/$TAG; log "$TAG: inicio (la PLE en GPU; su 1× la lleva en disco)"
if reinicia_1x EXTRA_ARGS- VLLM_PLE_MMAP=0; then
  kubectl logs -n llm deploy/ursu-1x > $T/$TAG/arranque.log 2>&1
  TOOLS=/home/user/bench-results/dgx-768/tools
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $R --model qwen3.8-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $T/$TAG/decode-agg.json > $T/$TAG/decode-agg.log 2>&1
  log "decode exit=$?"; tr '\r' '\n' < $T/$TAG/decode-agg.log | grep -E '^C= ' | cut -c1-120
  benchy $T/$TAG 0 65535
  log "dgx3: $(ssh dgx3 'free -g | sed -n 2p')"
else kubectl logs -n llm deploy/ursu-1x --previous --tail=60 > $T/$TAG/run.log 2>&1; fi
log "$TAG: fin"
TAG=c-perfil; mkdir -p $T/$TAG; log "$TAG: inicio"
ssh dgx3 'sudo -n rm -rf /srv/ursu-cache/1x/perfil-c'
if reinicia_1x VLLM_PLE_MMAP=1 EXTRA_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/xdg/perfil-c --profiler-config.torch_profiler_with_stack=false"; then
  python3 -I $D/prueba-c.py $R $T/$TAG/carga.json; log "carga exit=$?"; sleep 60
  ssh dgx3 'sudo -n tar -C /srv/ursu-cache/1x/perfil-c -cf - .' | tar -C $T/$TAG -xf -
  python3 -I $D/perfil-resumen.py $(find $T/$TAG -name "*.json*" ! -name carga.json | sort) > $T/$TAG/resumen.txt 2>&1; log "resumen exit=$?"
  head -30 $T/$TAG/resumen.txt
fi
log "$TAG: fin"
