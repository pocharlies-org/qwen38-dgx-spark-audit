#!/usr/bin/env bash
# DGX-852 (14-10): cierra la ventana con producción en 2 + 1. Requisitos: gitops #604 fusionada (ai → 8d1c46b, overlays/
# qwen38-ursu-2mas1-20261014) y el árbitro sin vision-deep (dgx-infra #1169) desplegado. Devuelve la auto-sync a root, que
# reescribe `ai` desde git (revisión, path y su auto-sync); la sync de ai corre sus hooks (PreSync pausa, PostSync
# llm-deploy-resume, que deja el árbitro en ready y el watcher repone qwen38). Sigue hasta los dos servidores sirviendo.
set -u
export KUBECONFIG=/home/user/.kube/config
O=/home/user/bench-results/dgx-847/ops
REV=8d1c46b549c1881391e9389e4383364ff4e8a43a; RUTA=overlays/qwen38-ursu-2mas1-20261014
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
fase() { kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["phase"], d.get("last_error"))'; }

git -C /home/user/k8s/k8s-gitops-pocharlies fetch -q origin deploy/prod
git -C /home/user/k8s/k8s-gitops-pocharlies show origin/deploy/prod:apps/ai.yaml | grep -q "targetRevision: $REV" \
  || { log "deploy/prod aún no lleva $REV (#604 sin fusionar): root resincronizaría la versión vieja"; exit 2; }
log "root: auto-sync de vuelta ($(cat $O/root-syncpolicy-antes.json))"
kubectl patch application -n argocd root --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $O/root-syncpolicy-antes.json)}}"
kubectl annotate application -n argocd root argocd.argoproj.io/refresh=normal --overwrite >/dev/null

fin=$(( $(date +%s) + 600 ))
until [ "$(kubectl get application -n argocd ai -o jsonpath='{.spec.source.targetRevision} {.spec.source.path} {.spec.syncPolicy.automated.selfHeal}')" = "$REV $RUTA true" ]; do
  [ "$(date +%s)" -lt $fin ] || { log "TOPE: root no reescribió ai"; exit 4; }
  sleep 10
done
log "ai reescrita por root: $REV $RUTA con auto-sync"

fin=$(( $(date +%s) + 5400 ))  # qwen38 tardaba ~33 min en arrancar con ~20 min sin log; el fork arrancó más rápido en la ventana
while [ "$(date +%s)" -lt $fin ]; do
  a=$(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status}/{.status.health.status}/{.status.operationState.phase}@{.status.operationState.syncResult.revision}')
  d=$(kubectl get deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
  h2=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark2:8000/health); h1=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark3:8000/health)
  p=$(fase)
  log "ai=${a:0:60} arb=$p deploys=[$d] health tp2=$h2 tp1=$h1"
  if [ "$h2" = 200 ] && [ "$h1" = 200 ] && [ "${p%% *}" = ready ] && [ "$d" = "1/1 1/1 1/1 " ]; then
    for u in http://spark2:8000 http://spark3:8000; do
      log "$u modelos=$(curl -s -m 10 $u/v1/models | jq -c '[.data[].id]') lambda=$(curl -s -m 10 $u/admin/refusal_lambda | head -c 200)"
    done
    kubectl get endpoints -n llm qwen38-flash-next qwen38-flash-next-head tooling qwen38-flash-next-tp1 --no-headers
    kubectl get deploy,svc -n llm vllm-vision-deep --no-headers 2>&1 | head -2
    log LISTO; exit 0
  fi
  sleep 60
done
log TOPE; exit 1
