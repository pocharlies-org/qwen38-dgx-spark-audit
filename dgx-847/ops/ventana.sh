#!/usr/bin/env bash
# DGX-847: abre la ventana. Espera a las descargas, desconecta ArgoCD (root antes que ai: root reharía la syncPolicy de ai),
# pausa el árbitro y la voz (Job llm-deploy-pause de DGX-820), para qwen38, suelta la caché y aplica las recetas de ursu.
# La vuelta es cierre.sh. Cada paso deja una línea con hora en la salida.
set -euo pipefail
export KUBECONFIG=/home/user/.kube/config
O=/home/user/bench-results/dgx-847/ops
PAUSE=/tmp/claude-1000/-home-dibanez-k8s/5d35c318-9261-4879-b52f-24fa3c4db837/scratchpad/glm64/pause.yaml
log() { echo "$(date -u +%H:%M:%SZ) $*"; }

log "espero a las descargas"
kubectl wait -n llm job/ursu-descarga-dgx1 job/ursu-descarga-dgx3 --for=condition=Complete --timeout=7200s
for j in dgx1 dgx3; do kubectl logs -n llm job/ursu-descarga-$j | grep '^OK'; done

log "ArgoCD: guardo y quito auto-sync de root y ai"
for a in root ai; do
  kubectl get application -n argocd $a -o jsonpath='{.spec.syncPolicy}' > $O/$a-syncpolicy-antes.json
  grep -q automated $O/$a-syncpolicy-antes.json || { log "ERROR: $a ya sin automated, no piso la copia"; exit 1; }
  kubectl patch application -n argocd $a --type json -p '[{"op":"remove","path":"/spec/syncPolicy/automated"}]'
done

log "pausa del árbitro y de la voz"
kubectl delete job -n llm llm-deploy-pause --ignore-not-found --wait=true
kubectl apply -f $PAUSE
kubectl wait -n llm job/llm-deploy-pause --for=condition=Complete --timeout=600s

log "qwen38 a 0"
kubectl scale deploy -n llm qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3 --replicas=0
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done

log "drop_caches en los tres Sparks"
for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; free -g | sed -n 2p'; done

log "recetas de ursu"
kubectl create configmap -n llm ursu-recipe --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap -n llm ursu-env-2x --from-env-file=$O/recipe/env-2x.env --from-env-file=$O/recipe/env-sparkrun-2x.env --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap -n llm ursu-env-1x --from-env-file=$O/recipe/env-1x.env --from-env-file=$O/recipe/env-sparkrun-1x.env --dry-run=client -o yaml | kubectl apply -f -
kubectl label cm -n llm ursu-recipe ursu-env-2x ursu-env-1x dgx-847=prueba --overwrite
kubectl apply -f $O/ursu.yaml
log "VENTANA ABIERTA"
