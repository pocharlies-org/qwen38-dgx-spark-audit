#!/usr/bin/env bash
# DGX-847: lanza la receta TP=3 en la ventana. Da por hecho la ventana YA abierta por ops/ventana.sh
# (ArgoCD root y ai sin auto-sync, Job llm-deploy-pause, qwen38 a 0, drop_caches) y solo crea los
# ConfigMaps de la 3x y aplica tp3/ursu-3x.yaml. La vuelta es la de siempre: ops/cierre.sh.
set -euo pipefail
export KUBECONFIG=/home/user/.kube/config
T=/home/user/bench-results/dgx-847/tp3
VIEW=/home/user/models-dgx847/tp3-view   # lo que produce convert-ursu-tp3.py en el x86
log() { echo "$(date -u +%H:%M:%SZ) $*"; }

# La vista se genera en el x86 (convert-ursu-tp3.py, sin tocar los Sparks) y de aqui se publica. Son ~120 GB
# a cada Spark: es escritura en su disco, asi que SOLO con la ventana cerrada. `--publicar` hace eso y para.
publicar() {
  for h in dgx dgx3; do
    base=/home/dibanerz/.cache/huggingface/tp3; [ $h = dgx3 ] && base=/srv/models/hf-ursu/tp3
    ssh $h "mkdir -p $base"
    for d in ursu-qwen38fn-tp3 ursu-mxfp8-tp3; do
      log "rsync $VIEW/$d -> $h:$base/$d"
      rsync -a --partial --info=stats2 $VIEW/$d/ $h:$base/$d/
      if ssh $h "cd $base/$d && sha256sum -c SHA256SUMS" | grep -qv ': OK$'; then
        log "SHA256SUMS no casa en $h:$base/$d"; exit 1
      fi
      log "  $h:$d verificado por sha256"
    done
  done
}
if [ "${1:-}" = "--publicar" ]; then publicar; exit 0; fi

log "compruebo la vista TP=3 en dgx1 (la sirve el PVC ro a dgx2) y en dgx3"
for h in dgx dgx3; do
  base=/home/dibanerz/.cache/huggingface/tp3; [ $h = dgx3 ] && base=/srv/models/hf-ursu/tp3
  ssh $h "test -f $base/ursu-qwen38fn-tp3/config.json && test -f $base/ursu-mxfp8-tp3/model.safetensors.index.json" \
    || { log "FALTA la vista TP=3 en $h:$base — publica:  $0 --publicar"; exit 1; }
  log "  $h ok"
done

log "ConfigMaps de la 3x"
kubectl create configmap -n llm ursu-recipe-3x --from-file=$T/recipe/serve-3x.sh --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap -n llm ursu-env-3x --from-env-file=$T/recipe/env-3x.env --from-env-file=$T/recipe/env-ring-3x.env --dry-run=client -o yaml | kubectl apply -f -
kubectl label cm -n llm ursu-recipe-3x ursu-env-3x dgx-847=prueba-3x --overwrite

log "tres rangos arriba"
kubectl apply -f $T/ursu-3x.yaml
kubectl -n llm rollout status deploy/ursu-3x-worker-dgx1 deploy/ursu-3x-worker-dgx3 --timeout=10s || true
log "TP=3 lanzado. El primer arranque es autotune FRIO (la semilla de la imagen esta claveada a TP=2):"
log "  seguir con  kubectl logs -n llm -l dgx-847=prueba-3x -f  y esperar 'b12x ready ...' + /health 200"
