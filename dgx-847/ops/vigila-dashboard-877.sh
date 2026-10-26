#!/usr/bin/env bash
# DGX-852/877 (14-10): el árbitro sin vision-deep (dgx-infra #1169, f958a911) tiene que estar corriendo ANTES de la sync de
# `ai` que borra vllm-vision-deep: con el código viejo, _enforce_exclusive_profile lanza «missing deployment».
# Sigue el build (push a master lo lanza; él mismo re-pinea los Deployments) y el rollout hasta ver el código nuevo en los pods.
set -u
export KUBECONFIG=/home/user/.kube/config
RUN=${1:-38614727017}; VIEJA=ci-9079555a
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
fin=$(( $(date +%s) + 2700 ))  # build: 1,4-7 min con caché, 71 min con arc saturado (durations.json)
while :; do
  s=$(gh run view $RUN -R pocharlies-org/dgx-infra --json status,conclusion --jq '"\(.status) \(.conclusion)"' 2>/dev/null)
  case "$s" in completed\ success) log "build $RUN ok"; break;; completed\ *) log "build $RUN FALLA: $s"; exit 1;; esac
  [ "$(date +%s)" -lt $fin ] || { log "TOPE build: $s"; exit 4; }
  sleep 30
done
fin=$(( $(date +%s) + 1200 ))
while :; do
  ok=1
  for d in dgx-dashboard-backend dgx-dashboard-worker; do
    st=$(kubectl get deploy -n control-nexus $d -o json | jq -r '"\(.spec.template.spec.containers[0].image|split(":")[1]|split("@")[0]) \(.status.updatedReplicas // 0)/\(.spec.replicas)/\(.status.availableReplicas // 0) \(.status.unavailableReplicas // 0)"')
    log "$d $st"
    set -- $st
    [ "$1" != $VIEJA ] && [ "$2" = 1/1/1 ] && [ "$3" = 0 ] || ok=0
  done
  if [ $ok = 1 ]; then
    n=$(kubectl exec -n control-nexus deploy/dgx-dashboard-backend -c backend -- grep -c '"vision-deep"' /app/compute_mode.py 2>/dev/null)
    m=$(kubectl exec -n control-nexus deploy/dgx-dashboard-worker -- grep -c '"vision-deep"' /app/compute_mode.py 2>/dev/null)
    log "\"vision-deep\" en compute_mode.py: backend=$n worker=$m"
    [ "$n" = 0 ] && [ "$m" = 0 ] && { log LISTO; exit 0; }
  fi
  [ "$(date +%s)" -lt $fin ] || { log "TOPE rollout"; exit 4; }
  sleep 30
done
