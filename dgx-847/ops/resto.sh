#!/usr/bin/env bash
# DGX-847: benchy y calidad en las dos recetas sin tope de hora (sustituye a orquesta.sh desde las 20:45Z), y al acabar cierre.sh.
D=/home/user/bench-results/dgx-847; O=$D/ops
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
declare -A URL=([2x]=http://spark2:8000 [1x]=http://spark3:8000) DEP=([2x]=ursu-2x-head [1x]=ursu-1x)
for a in 2x 1x; do
  ( log "$a: calidad (benchy ya hecho; sin tope de hora)"
    RES=${URL[$a]} DEPLOY=${DEP[$a]} ARM=$a BASE=$D/$a MODEL=qwen3.8-flash-next DEADLINE=$(( $(date +%s) + 12*3600 )) BLACKOUT=12:00-12:01 SKIP_BENCHY=1 \
      KUBECONFIG=/home/user/.kube/config python3 $D/launch-benchy-quality.py > $D/$a/run-quality.log 2>&1
    log "$a: calidad exit=$? ($(head -2 $D/$a/DONE-benchy-quality | tr '\n' ' '))" ) >> $D/chain-$a.log 2>&1 &
done
wait
log "cierre de la ventana"
$O/cierre.sh > $O/cierre.log 2>&1; log "cierre exit=$? ($(tail -1 $O/cierre.log))"
