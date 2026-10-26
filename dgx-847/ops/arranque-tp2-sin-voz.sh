#!/usr/bin/env bash
# DGX-852 (14-10, decisión de Dani): el TP=2 no cabe en dgx2 durante el arranque con stt-turbo arriba (la prueba arrancó con la
# voz parada: pico de 1,5 GiB libres; con stt-turbo, 0,3 y se cuelga al reservar el KV). Arranque con la voz parada, a mano y
# con los mismos pasos que los hooks de `ai`: pausa del árbitro (sync-paused) → stt-turbo a 0 → pods del TP=2 de nuevo →
# TP=2 sano → stt-turbo a 1 → árbitro a ready (lo que hace llm-deploy-resume).
set -u
export KUBECONFIG=/home/user/.kube/config
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
estado() { kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'; }
escribe() { kubectl patch cm -n comfyui gpu-arbiter-state --type merge -p "$(python3 -c 'import json,sys; print(json.dumps({"data": {"compute_mode": sys.argv[1]}}))' "$1")" >/dev/null; }
libre() { timeout 20 ssh -o ConnectTimeout=8 $1 'awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo'; }

st=$(estado); log "árbitro antes: $(echo "$st" | jq -c '{phase,last_error}')"
escribe "$(echo "$st" | jq -c --arg at "$(date -u +%FT%TZ)" '.phase="sync-paused" | .sync_pause={by:"dgx-852-arranque-tp2",at:$at}')"
log "árbitro en pausa: $(estado | jq -c '{phase,sync_pause}')"

kubectl scale deploy -n llm stt-turbo --replicas=0 >/dev/null
until [ -z "$(kubectl get pods -n llm -l app=stt-turbo --no-headers 2>/dev/null)" ]; do sleep 3; done
log "stt-turbo parado"

kubectl delete pod -n llm -l 'app in (qwen38-flash-next-head,qwen38-flash-next-worker)' --wait=true >/dev/null
for h in dgx dgx2; do timeout 30 ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
log "pods del TP=2 de nuevo; libre dgx2=$(libre dgx2) GiB dgx1=$(libre dgx) GiB"

fin=$(( $(date +%s) + 2400 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark2:8000/health)" = 200 ]; do
  [ "$(date +%s)" -lt $fin ] || { log "TOPE: TP=2 sin /health; árbitro sigue en pausa y stt-turbo a 0"; exit 4; }
  log "TP=2 arrancando: $(kubectl get pods -n llm -l 'app in (qwen38-flash-next-head,qwen38-flash-next-worker)' --no-headers | awk '{print $1":"$3":"$4}' | tr '\n' ' ') libre dgx2=$(libre dgx2) GiB"
  sleep 60
done
log "TP=2 sano: $(curl -s -m 10 http://spark2:8000/v1/models | jq -c '[.data[].id]'); libre dgx2=$(libre dgx2) GiB"

kubectl scale deploy -n llm stt-turbo --replicas=1 >/dev/null
fin=$(( $(date +%s) + 900 ))
until [ "$(kubectl get deploy -n llm stt-turbo -o jsonpath='{.status.readyReplicas}')" = 1 ]; do
  [ "$(date +%s)" -lt $fin ] || { log "TOPE: stt-turbo no vuelve; árbitro sigue en pausa"; exit 4; }
  sleep 10
done
log "stt-turbo de vuelta; libre dgx2=$(libre dgx2) GiB"

escribe "$(estado | jq -c 'del(.sync_pause) | .phase="ready" | .blockers=[] | .last_error=null')"
log "árbitro a ready"
for i in $(seq 1 10); do
  sleep 30
  p=$(estado | jq -c '{phase,last_error}'); d=$(kubectl get deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 stt-turbo omnivoice-tts-nvfp4 -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
  log "árbitro=$p deploys=[$d] health tp2=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark2:8000/health) tp1=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark3:8000/health) libre dgx2=$(libre dgx2) GiB"
done
log FIN
