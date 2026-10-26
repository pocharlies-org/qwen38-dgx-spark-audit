#!/usr/bin/env bash
# DGX-852: MTP k=2, 3 y 4 en el residente con la valla puesta (default_plan=alibaba en model-routing, 13-10 ~10:00Z; Dani dijo sí).
# k=2 ya está arriba (vuelta-residente.sh); k=3 y k=4 con pausa/resume del árbitro. Al acabar hay que QUITAR LA VALLA (panel
# internal.example/model-routing → local) y fijar la k buena por PR con ArgoCD de vuelta.
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; RES=http://tailnet-node:8000
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
BENCHY=/home/user/bin/benchy-venv/bin/llama-benchy; TOOLS=/home/user/bench-results/dgx-768/tools
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
ajenas() { curl -s -m 5 $RES/metrics | awk '/^vllm:num_requests_running/{s+=$2} END{print int(s)}'; }
espera_vacio() {  # con la valla, las sesiones ya ligadas a local acaban solas: espera a 0 peticiones ajenas (tope 20 min)
  local t0=$(date +%s) n
  while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do n=$(ajenas); [ "$n" = 0 ] && { sleep 20; [ "$(ajenas)" = 0 ] && return 0; }; sleep 20; done
  log "sigue habiendo tráfico ajeno ($(ajenas) en curso) tras 20 min: mido igual; la sonda lo marca"
}
for k in 2 3 4; do
  M=$D/mtp/k$k; mkdir -p $M
  if [ $k != 2 ]; then  # el residente está arriba: pausa del árbitro y a 0 antes de tocar el env (TP=3 no se reinicia por partes)
    hook llm-deploy-pause llm-deploy-pause-mtp$k
    kubectl scale deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 --replicas=0
    while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
    for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
  fi
  if [ $k != 2 ]; then
  SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$k,\"disable_eagle_block_drop\":true,\"num_speculative_tokens_per_batch_size\":[[1,16,$k]],\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"probabilistic\"}"
  for d in qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3; do kubectl set env -n llm deploy/$d SPECULATIVE_CONFIG="$SPEC" >/dev/null; done
  log "MTP k=$k: resume del árbitro"
  hook llm-deploy-resume llm-deploy-resume-mtp$k
  fi
  listo || { kubectl logs -n llm deploy/qwen38-flash-next-head --tail=200 > $M/arranque.log 2>&1; log "MTP k=$k: no arranca, sigo con la siguiente"; continue; }
  kubectl logs -n llm deploy/qwen38-flash-next-head > $M/arranque.log 2>&1
  log "MTP k=$k: $(grep -oE 'attention block size to [0-9]+|GPU KV cache size: [0-9,]+ tokens|num_speculative_tokens.{0,4}' $M/arranque.log | sort -u | paste -sd' ')"
  espera_vacio; log "MTP k=$k: mido (ajenas en curso: $(ajenas))"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $M/decode-agg.json > $M/decode-agg.log 2>&1
  log "MTP k=$k: decode exit=$?"; tr '\r' '\n' < $M/decode-agg.log | grep -E '^C= ' | cut -c1-120
  $BENCHY --base-url $RES/v1 --model qwen38-flash-next --served-model-name qwen38-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 4 8 --runs 3 --enable-prefix-caching --format csv --save-result $M/medium.csv > $M/benchy.log 2>&1
  log "MTP k=$k: benchy exit=$?"
done
log "MTP: fin (residente con k=4, valla PUESTA y ArgoCD en pausa: quitar la valla y fijar la k por PR)"
