#!/usr/bin/env bash
# DGX-847 tras G2, en orden:
#  1. E2: su 2× con el estado SSM en BF16 y los kernels GDN estándar (triton) en vez de los b12x, que exigen FP32 (prueba E).
#     El GDN es el 1-2,5 % del tiempo de GPU (prueba C): se pierde poco al dejar b12x ahí y se gana el bloque de caché pequeño.
#  2. Fuera la 2× (no está en ArgoCD).
#  3. Nuestro residente vuelve con MTP k=2, luego 3 y 4 (lo pidió Dani). Ventana: root y ai sin auto-sync; las réplicas las mueve
#     el árbitro con los jobs de pausa/resume (el PostSync de la sync de las 07:46Z no corrió). En cada k: decode v4
#     (c1/4/8/10 ×3) y benchy medium; la sonda marca como contaminado lo que coincida con tráfico real.
#  Queda con k=4 y ArgoCD en pausa: la k buena se fija después por PR.
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; R=http://spark2:8000; RES=http://tailnet-node:8000
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
. $O/espera-2x.sh
BENCHY=/home/user/bin/benchy-venv/bin/llama-benchy; TOOLS=/home/user/bench-results/dgx-768/tools

# ---- 1. E2
T=$D/pruebas/e2-bf16-gdn-triton; mkdir -p $T
log "E2: inicio (estado SSM en BF16 + GDN triton para prefill y decode)"
reinicia_2x VLLM_PLE_MMAP- VLLM_PLE_MMAP_WILLNEED_MAX- VLLM_PLE_MMAP_KEEPALIVE_MS- VLLM_PLE_MMAP_PREFILL_WILLNEED- VLLM_GDN_DEFERRED_CHECKPOINTS=0 \
  EXTRA_ARGS='--mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel triton --additional-config {"gdn_prefill_backend":"triton","gdn_decode_kernel":"triton"}'
ok=0; t0=$(date +%s); sleep 20
while [ $(( $(date +%s) - t0 )) -lt 2700 ]; do  # un fallo de configuración se repite: al primer reinicio se da por fallida
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R/health)" = 200 ] && { ok=1; break; }
  kubectl get pods -n llm -l "$SEL_2X" -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | grep -qE '[1-9]' && break
  sleep 15
done
kubectl logs -n llm deploy/ursu-2x-head > $T/arranque.log 2>&1 || kubectl logs -n llm deploy/ursu-2x-head --previous > $T/arranque.log 2>&1
grep -E 'attention block size|GPU KV cache size|mamba_ssm_dtype|gdn.*(backend|kernel)' $T/arranque.log | cut -c1-200 | sort -u | head -6
if [ $ok = 1 ]; then
  log "E2: 2× lista en $(( ($(date +%s) - t0) / 60 )) min"
  python3 -I $D/prueba-a-cache.py $R $T/a-cache-ctx65535.json 65535 | sed -u 's/^/E2 A: /'
  $BENCHY --base-url $R/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 8 --runs 3 --enable-prefix-caching --format csv --save-result $T/medium.csv > $T/run.log 2>&1
  log "E2: benchy exit=$?"
else
  { echo "no arranca: $(grep -E 'ValueError|RuntimeError|Error:' $T/arranque.log | grep -v 'APIServer pid=1)     ' | head -2 | cut -c1-300)"; cat $T/arranque.log; } > $T/run.log
  log "E2: $(head -1 $T/run.log)"
fi
log "E2: fin"

# ---- 2. fuera la 2×
log "fuera la 2×"
kubectl delete deploy,cm -n llm -l dgx-847=prueba --wait=true
while kubectl get pods -n llm --no-headers | grep -q '^ursu-'; do sleep 5; done
for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done

# ---- 3. residente con MTP k=2,3,4
hook() {  # $1 = job de ArgoCD a clonar (llm-deploy-pause|llm-deploy-resume), $2 = nombre nuevo; espera a que acabe
  kubectl get job -n llm $1 -o json | python3 -c '
import json,sys; j=json.load(sys.stdin); t=j["spec"]["template"]; l=t["metadata"].get("labels",{})
for k in ("controller-uid","batch.kubernetes.io/controller-uid","job-name","batch.kubernetes.io/job-name"): l.pop(k,None)
print(json.dumps({"apiVersion":"batch/v1","kind":"Job","metadata":{"name":sys.argv[1],"namespace":"llm","labels":{"app":"llm-deploy-hooks"}},
 "spec":{"backoffLimit":j["spec"].get("backoffLimit",2),"ttlSecondsAfterFinished":86400,"template":t}}))' $2 | kubectl create -f - >/dev/null
  kubectl wait -n llm job/$2 --for=condition=Complete --timeout=1200s >/dev/null && log "job $2 completo" || log "job $2 NO completo"
}
listo() {  # residente 1/1 ×3, árbitro ready, /health 200 (tope 60 min)
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 3600 ]; do
    p=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
    d=$(kubectl get deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
    h=$(curl -s -o /dev/null -w '%{http_code}' -m 5 $RES/health || true)
    [ "$h" = 200 ] && [ "$p" = ready ] && [ "$d" = "1/1 1/1 1/1 " ] && { log "residente listo en $(( ($(date +%s) - t0) / 60 )) min"; return 0; }
    sleep 30
  done; log "residente sin listo en 60 min (arb=$p deploys=[$d] health=$h)"; return 1
}
log "ArgoCD: fuera auto-sync de root y ai (copia en *-syncpolicy-mtp.json)"
for a in root ai; do
  kubectl get application -n argocd $a -o jsonpath='{.spec.syncPolicy}' > $O/$a-syncpolicy-mtp.json
  grep -q automated $O/$a-syncpolicy-mtp.json && kubectl patch application -n argocd $a --type json -p '[{"op":"remove","path":"/spec/syncPolicy/automated"}]'
done
for k in 2 3 4; do
  M=$D/mtp/k$k; mkdir -p $M
  if [ $k != 2 ]; then  # el residente está arriba: pausa del árbitro y a 0 antes de tocar el env (TP=3 no se reinicia por partes)
    hook llm-deploy-pause llm-deploy-pause-mtp$k
    kubectl scale deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 --replicas=0
    while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
    for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
  fi
  SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$k,\"disable_eagle_block_drop\":true,\"num_speculative_tokens_per_batch_size\":[[1,16,$k]],\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"probabilistic\"}"
  for d in qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3; do kubectl set env -n llm deploy/$d SPECULATIVE_CONFIG="$SPEC" >/dev/null; done
  log "MTP k=$k: resume del árbitro"
  hook llm-deploy-resume llm-deploy-resume-mtp$k
  listo || { kubectl logs -n llm deploy/qwen38-flash-next-head --tail=200 > $M/arranque.log 2>&1; log "MTP k=$k: no arranca, sigo con la siguiente"; continue; }
  kubectl logs -n llm deploy/qwen38-flash-next-head > $M/arranque.log 2>&1
  log "MTP k=$k: $(grep -oE 'attention block size to [0-9]+|GPU KV cache size: [0-9,]+ tokens|num_speculative_tokens.{0,4}' $M/arranque.log | sort -u | paste -sd' ')"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $M/decode-agg.json > $M/decode-agg.log 2>&1
  log "MTP k=$k: decode exit=$?"; tr '\r' '\n' < $M/decode-agg.log | grep -E '^C= ' | cut -c1-120
  $BENCHY --base-url $RES/v1 --model qwen38-flash-next --served-model-name qwen38-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 4 8 --runs 3 --enable-prefix-caching --format csv --save-result $M/medium.csv > $M/benchy.log 2>&1
  log "MTP k=$k: benchy exit=$?"
done
log "MTP: fin (residente con k=4, ArgoCD de root y ai en pausa hasta fijar la k por PR)"
