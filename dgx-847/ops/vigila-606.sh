#!/usr/bin/env bash
# DGX-898: gitops #606 (ai -> 72417e5) hasta produccion: merge, sync de ai con head y worker del TP=2 reiniciados a la vez,
# el worker enganchado a la sesion NUEVA del head (log worker-sigue-head), TP=2 Ready, voz arriba, arbitro ready y TP=1 intacto.
set -u
export KUBECONFIG=~/.kube/config
REV=72417e52588482ed41a9f76d109adad4196117e9
SEL='app in (qwen38-flash-next-head,qwen38-flash-next-worker)'
~/.claude/skills/pr-watcher/scripts/pr-watch.sh --repo pocharlies-org/k8s-gitops-pocharlies --pr 606 || { echo "pr-watch rc=$?: no sigo"; exit 1; }
kubectl annotate application -n argocd root argocd.argoproj.io/refresh=normal --overwrite >/dev/null
fin=$(( $(date +%s) + 4800 )); prev=""; prevlog=""
while [ "$(date +%s)" -lt "$fin" ]; do
  spec=$(kubectl get application -n argocd ai -o jsonpath='{.spec.source.targetRevision}')
  op=$(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase} {.status.operationState.syncResult.revision}')
  pods=$(kubectl get pods -n llm -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.labels.app}:{.status.phase}:{.status.containerStatuses[0].ready}:r{.status.containerStatuses[0].restartCount} {end}' 2>/dev/null)
  jobs=$(kubectl get job -n llm llm-deploy-pause llm-deploy-resume -o jsonpath='{range .items[*]}{.metadata.name}:{.status.succeeded}/{.status.failed} {end}' 2>/dev/null)
  linea="spec=${spec:0:8} op=$op jobs=$jobs pods=$pods"
  [ "$linea" != "$prev" ] && echo "$(date -u +%T) $linea"; prev=$linea
  wl=$(kubectl logs -n llm -l app=qwen38-flash-next-worker --tail=300 2>/dev/null | grep 'worker-sigue-head' | tail -1)
  [ -n "$wl" ] && [ "$wl" != "$prevlog" ] && echo "$(date -u +%T) worker: $wl"; prevlog=$wl
  [ "$op" = "Succeeded $REV" ] && break
  case "$op" in Failed*|Error*) [ "${op#* }" = "$REV" ] && { echo "sync de ai FALLIDA"; break; } ;; esac
  sleep 30
done
echo "--- resume:"; kubectl logs -n llm job/llm-deploy-resume --tail=6 2>&1
echo "--- arbitro: $(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | jq -c '{phase,effective_mode,blockers,last_error,sync_pause}')"
echo "--- voz: $(kubectl get deploy -n llm omnivoice-tts-nvfp4 stt-turbo -o jsonpath='{range .items[*]}{.metadata.name}={.status.readyReplicas}/{.spec.replicas} {end}')"
echo "--- pods (antes / ahora):"; cat /tmp/claude-1000/dgx852/pods-antes-606.txt
kubectl get pods -n llm -l 'app in (qwen38-flash-next-head,qwen38-flash-next-worker,qwen38-flash-next-worker-dgx3)' -o custom-columns=N:.metadata.name,UID:.metadata.uid,START:.status.startTime,REV:.metadata.annotations.config\.k8s\.e-dani\.com/launch-revision,RST:.status.containerStatuses[0].restartCount --no-headers
echo "--- worker-sigue-head:"; kubectl logs -n llm -l app=qwen38-flash-next-worker --tail=2000 2>/dev/null | grep 'worker-sigue-head' | tail -5
for h in spark2 spark3; do echo "health $h: $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://$h:8000/health)"; done
echo "--- ai: $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status} {.status.health.status} {.status.sync.revision}')"
