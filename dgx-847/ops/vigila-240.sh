#!/usr/bin/env bash
# DGX-897: k8s-litellm #240 hasta produccion: merge, sync de la app litellm al SHA y rollout de los pods.
set -u
export KUBECONFIG=~/.kube/config
~/.claude/skills/pr-watcher/scripts/pr-watch.sh --repo pocharlies-org/k8s-litellm-pocharlies --pr 240 || { echo "pr-watch rc=$?: no sigo"; exit 1; }
REV=$(gh pr view 240 -R pocharlies-org/k8s-litellm-pocharlies --json mergeCommit -q .mergeCommit.oid)
echo "merge $REV"
kubectl annotate application -n argocd litellm argocd.argoproj.io/refresh=normal --overwrite >/dev/null
fin=$(( $(date +%s) + 1500 ))
while [ "$(date +%s)" -lt "$fin" ]; do
  rev=$(kubectl get application -n argocd litellm -o jsonpath='{.status.sync.revision}')
  st=$(kubectl rollout status deploy/litellm -n litellm --timeout=5s 2>&1 | tail -1)
  ann=$(kubectl get deploy -n litellm litellm -o jsonpath='{.spec.template.metadata.annotations.config\.k8s\.e-dani\.com/revision}')
  echo "$(date -u +%T) app=${rev:0:8} ann=$ann rollout=$st"
  [ "$rev" = "$REV" ] && [ "$ann" = "config-3acc7c06b354" ] && echo "$st" | grep -q "successfully rolled out" && { echo LISTO; break; }
  sleep 30
done
kubectl get pods -n litellm -l app=litellm -o custom-columns=N:.metadata.name,S:.status.phase,R:.status.containerStatuses[0].ready,AGE:.metadata.creationTimestamp --no-headers 2>/dev/null
