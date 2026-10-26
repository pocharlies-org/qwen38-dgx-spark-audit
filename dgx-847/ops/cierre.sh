#!/usr/bin/env bash
# DGX-847: cierra la ventana. Quita las recetas de ursu, devuelve la auto-sync a root y ai, y lanza una sync de ai para que
# corran sus hooks (PreSync pausa, que acepta sync-paused; PostSync llm-deploy-resume, que deja el árbitro en ready y el
# watcher repone qwen38). Sigue hasta qwen38 TP=3 listo, árbitro en ready y /health 200. Los pesos de ursu se quedan.
set -euo pipefail
export KUBECONFIG=/home/user/.kube/config
O=/home/user/bench-results/dgx-847/ops
log() { echo "$(date -u +%H:%M:%SZ) $*"; }

# las pruebas A-C de DGX-847 (una a una, en la 2×) retienen el cierre mientras exista este fichero
while [ -e $O/NO-CERRAR ]; do sleep 30; done
log "Claude Code contra cada receta (tras la batería, para no contaminarla)"
for a in 2x:http://spark2:8000 1x:http://spark3:8000; do
  n=${a%%:*}; u=${a#*:}; [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $u/health || true)" = 200 ] || { log "$n sin /health, sin prueba de Claude Code"; continue; }
  W=$(mktemp -d /tmp/dgx847-cc-XXXX); echo "hola desde DGX-847" > $W/entrada.txt
  # --setting-sources project: sin el settings.json de usuario, que fija ANTHROPIC_BASE_URL y manda el CLI a LiteLLM
  ( cd $W && ANTHROPIC_BASE_URL=$u ANTHROPIC_API_KEY=dummy timeout 300 claude -p --setting-sources project --model qwen3.8-flash-next \
      --allowedTools "Read Write Bash(ls:*)" --permission-mode acceptEdits --output-format json \
      "Lee entrada.txt, escribe su contenido en mayúsculas en salida.txt y luego lista el directorio con ls." ) \
    > /home/user/bench-results/dgx-847/$n/claude-code.json 2>&1 || true
  log "$n claude-code: salida.txt=$(cat $W/salida.txt 2>/dev/null || echo NO)"
done

log "fuera las recetas de ursu"
kubectl delete deploy,cm,job -n llm -l dgx-847=prueba --wait=true
while kubectl get pods -n llm --no-headers | grep -q '^ursu-'; do sleep 5; done
for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done

log "ArgoCD: auto-sync de vuelta (ai antes que root) y sync de ai"
for a in ai root; do kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $O/$a-syncpolicy-antes.json)}}"; done
rev=$(kubectl get application -n argocd ai -o jsonpath='{.spec.source.targetRevision}')
kubectl patch application -n argocd ai --type merge -p "{\"operation\":{\"initiatedBy\":{\"username\":\"dgx-847-cierre\"},\"sync\":{\"revision\":\"$rev\"}}}"

fin=$(( $(date +%s) + 5400 ))
while [ "$(date +%s)" -lt $fin ]; do
  a=$(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status}/{.status.health.status}/{.status.operationState.phase}')
  p=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
  d=$(kubectl get deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
  h=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://tailnet-node:8000/health || true)
  log "ai=$a arb=$p deploys=[$d] health=$h"
  [ "$h" = 200 ] && [ "$p" = ready ] && [ "$d" = "1/1 1/1 1/1 " ] && { log LISTO; exit 0; }
  sleep 60
done; log TOPE; exit 1
