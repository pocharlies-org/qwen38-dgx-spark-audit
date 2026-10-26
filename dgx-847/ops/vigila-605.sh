#!/usr/bin/env bash
# DGX-852: gitops #605 (ai -> 4a422b8) hasta produccion: merge, sync de ai con los hooks nuevos, voz arriba, arbitro ready,
# TP=2 y TP=1 sin reiniciar (mismos UID que antes) y /health 200 en los dos.
set -u
export KUBECONFIG=~/.kube/config
REV=4a422b81d931440e0c8d9e38e94c4f63eac2e187
~/.claude/skills/pr-watcher/scripts/pr-watch.sh --repo pocharlies-org/k8s-gitops-pocharlies --pr 605 || { echo "pr-watch rc=$?: no sigo"; exit 1; }
kubectl annotate application -n argocd root argocd.argoproj.io/refresh=normal --overwrite >/dev/null
fin=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$fin" ]; do
  spec=$(kubectl get application -n argocd ai -o jsonpath='{.spec.source.targetRevision}')
  op=$(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase} {.status.operationState.syncResult.revision}')
  echo "$(date -u +%T) spec=${spec:0:8} op=$op jobs=$(kubectl get job -n llm llm-deploy-pause llm-deploy-resume -o jsonpath='{range .items[*]}{.metadata.name}:{.status.succeeded}/{.status.failed} {end}' 2>/dev/null)"
  [ "$op" = "Succeeded $REV" ] && break
  case "$op" in Failed*|Error*) [ "${op#* }" = "$REV" ] && { echo "sync de ai FALLIDA"; break; } ;; esac
  sleep 30
done
echo "--- resume:"; kubectl logs -n llm job/llm-deploy-resume --tail=5 2>&1
echo "--- arbitro: $(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | jq -c '{phase,effective_mode,blockers,last_error,sync_pause}')"
echo "--- voz: $(kubectl get deploy -n llm omnivoice-tts-nvfp4 stt-turbo -o jsonpath='{range .items[*]}{.metadata.name}={.status.readyReplicas}/{.spec.replicas} {end}')"
echo "--- pods (antes / ahora):"; cat /tmp/claude-1000/dgx852/pods-antes-605.txt
kubectl get pods -n llm -l 'app in (qwen38-flash-next-head,qwen38-flash-next-worker,qwen38-flash-next-worker-dgx3)' -o custom-columns=N:.metadata.name,UID:.metadata.uid,START:.status.startTime --no-headers
for h in spark2 spark3; do echo "health $h: $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://$h:8000/health)"; done
echo "--- ai: $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status} {.status.health.status} {.status.sync.revision}')"
