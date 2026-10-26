#!/usr/bin/env bash
# DGX-899: dgx-infra#1181 (7f60c24b) hasta produccion: build-dashboard-backend del push, re-pin en k8s/control-nexus,
# rollout del dgx-dashboard-backend con ci-7f60c24b y el watcher del arbitro vivo sin errores de la guarda nueva.
set -u
export KUBECONFIG=~/.kube/config
RUN=38648823134; SHA=7f60c24bd1608ff75445d1ba1b7a5c89b663ca71
gh run watch "$RUN" -R pocharlies-org/dgx-infra --exit-status --interval 60 >/dev/null 2>&1
echo "build: $(gh run view $RUN -R pocharlies-org/dgx-infra --json conclusion,updatedAt -q '.conclusion+" "+.updatedAt')"
[ "$(gh run view $RUN -R pocharlies-org/dgx-infra --json conclusion -q .conclusion)" = success ] || exit 1
kubectl annotate application -n argocd control-nexus argocd.argoproj.io/refresh=normal --overwrite >/dev/null 2>&1
fin=$(( $(date +%s) + 1200 ))
until kubectl get deploy -n control-nexus dgx-dashboard-backend -o jsonpath='{.spec.template.spec.containers[0].image}' | grep -q "ci-$SHA"; do
  [ "$(date +%s)" -gt "$fin" ] && { echo "TOPE: el backend no lleva ci-$SHA"; exit 4; }; sleep 30; done
kubectl rollout status -n control-nexus deploy/dgx-dashboard-backend --timeout=900s
sleep 120  # dos minutos de watcher (vuelta cada 5 s) antes de mirar el log
echo "--- log del backend (arbitro):"
kubectl logs -n control-nexus deploy/dgx-dashboard-backend --since=3m 2>&1 | grep -iE 'DGX-899|voice|voz|compute.?mode|Traceback|Error' | tail -10
echo "--- arbitro: $(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | jq -c '{phase,effective_mode,voz_parada_por_tp2}')"
echo "--- voz: $(kubectl get deploy -n llm omnivoice-tts-nvfp4 stt-turbo -o jsonpath='{range .items[*]}{.metadata.name}={.status.readyReplicas}/{.spec.replicas} {end}')"
