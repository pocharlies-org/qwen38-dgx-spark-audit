#!/usr/bin/env bash
# DGX-906: k8s-litellm#242 hasta produccion: puerta LISTA -> merge -> rollout de litellm con config-ff01ac87e879 ->
# el codigo vivo resuelve el TP=1 a su tarjeta -> una fila del TP=1 con server_id en /api/activity (si hay trafico, 20 min).
set -u
export KUBECONFIG=~/.kube/config
/home/user/bench-results/dgx-847/ops/espera-lista.sh pocharlies-org/k8s-litellm-pocharlies 242 2400 || exit $?
fin=$(( $(date +%s) + 900 ))
until kubectl get deploy -n litellm litellm -o jsonpath='{.spec.template.metadata.annotations.config\.k8s\.e-dani\.com/revision}' | grep -q config-ff01ac87e879; do
  [ "$(date +%s)" -gt "$fin" ] && { echo "TOPE: litellm sin config-ff01ac87e879"; exit 4; }; sleep 20; done
kubectl rollout status -n litellm deploy/litellm --timeout=900s || exit 1
P=$(kubectl get pods -n litellm -o name | grep -E 'pod/litellm-[0-9a-f]+-' | head -1)
echo "vivo: $(kubectl exec -n litellm $P -c litellm -- sh -c 'cd "$(dirname "$(find / -name active_request_tracking.py -path "*config*" 2>/dev/null | head -1)")" && python3 -c "import active_request_tracking as m; print(m.resolve_server_id(None, \"http://qwen38-flash-next-tp1.llm.svc.cluster.local:8000/v1\"))"' 2>&1 | tail -1)"
fin=$(( $(date +%s) + 1200 ))
while [ "$(date +%s)" -lt "$fin" ]; do
  r=$(kubectl exec -n control-nexus deploy/dgx-dashboard-backend -c backend -- python3 -c "
import urllib.request,json
d=json.load(urllib.request.urlopen('http://127.0.0.1:9002/api/activity',timeout=15))
print(sorted({str(x.get('server_id')) for x in d.get('active_requests',[]) if x.get('source')=='litellm'}))" 2>/dev/null)
  case "$r" in *qwen38-flash-next-worker-dgx3*) echo "$(date -u +%T) fila del TP=1 en su tarjeta: $r"; exit 0;; esac
  sleep 15
done
echo "sin trafico del TP=1 en 20 min: no se pudo ver una fila viva (ultimo: $r)"; exit 3
