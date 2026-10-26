#!/usr/bin/env bash
# DGX-852: control k=1 medido hoy en las mismas condiciones (valla puesta) tras k=2/3/4; la base de DGX-817 es de otro día y la
# 1.ª respuesta a 64K con caché salió peor con k=2 y k=3. k=1 es lo que hay en git: al acabar se devuelve la auto-sync de
# root y ai (sin diferencias en el env). La VALLA se quita a mano después (panel model-routing → local).
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; RES=http://tailnet-node:8000
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
BENCHY=/home/user/bin/benchy-venv/bin/llama-benchy; TOOLS=/home/user/bench-results/dgx-768/tools
until grep -q 'MTP: fin' $O/vuelta.log; do sleep 30; done
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
k=1; M=$D/mtp/k1-control; mkdir -p $M
hook llm-deploy-pause llm-deploy-pause-mtp1
kubectl scale deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 --replicas=0
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
SPEC='{"method":"mtp","num_speculative_tokens":1,"disable_eagle_block_drop":true,"num_speculative_tokens_per_batch_size":[[1,16,1]],"rejection_sample_method":"block","draft_sample_method":"probabilistic"}'
for d in qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3; do kubectl set env -n llm deploy/$d SPECULATIVE_CONFIG="$SPEC" >/dev/null; done
log "MTP k=1 (control): resume del árbitro"
hook llm-deploy-resume llm-deploy-resume-mtp1
if listo; then
  kubectl logs -n llm deploy/qwen38-flash-next-head > $M/arranque.log 2>&1
  espera_vacio; log "MTP k=1 (control): mido (ajenas en curso: $(ajenas))"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $M/decode-agg.json > $M/decode-agg.log 2>&1
  log "MTP k=1 (control): decode exit=$?"; tr '\r' '\n' < $M/decode-agg.log | grep -E '^C= ' | cut -c1-120
  $BENCHY --base-url $RES/v1 --model qwen38-flash-next --served-model-name qwen38-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 4 8 --runs 3 --enable-prefix-caching --format csv --save-result $M/medium.csv > $M/benchy.log 2>&1
  log "MTP k=1 (control): benchy exit=$?"
fi
log "ArgoCD: auto-sync de vuelta (ai antes que root), env igual que git"
for a in ai root; do kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $O/$a-syncpolicy-mtp.json)}}"; done
sleep 90; log "ai: $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status}/{.status.health.status}') · residente: $(curl -s -o /dev/null -w '%{http_code}' -m 5 $RES/health)"
log "MTP control: fin (falta quitar la VALLA)"
