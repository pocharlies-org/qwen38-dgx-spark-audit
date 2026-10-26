#!/usr/bin/env bash
# DGX-847: después de ventana.sh. Espera el arranque de las dos recetas (tope 60 min), lanza chain.sh en las que den /health,
# y cierra la ventana (cierre.sh) cuando acaban las dos o, como tarde, a las 00:15Z, para que el residente esté de vuelta
# antes de las 03:00 Madrid. La calidad se corta a las 00:05Z (DEADLINE).
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
CIERRE=$(date -u -d 'today 00:15' +%s); [ "$CIERRE" -lt "$(date +%s)" ] && CIERRE=$(date -u -d 'tomorrow 00:15' +%s)
export DEADLINE=$((CIERRE - 600))

until grep -qE 'VENTANA ABIERTA|ERROR' $O/ventana.log 2>/dev/null || ! pgrep -f '[v]entana.sh' >/dev/null; do sleep 10; done
grep -q 'VENTANA ABIERTA' $O/ventana.log || { log "ventana.sh no abrió la ventana: $(tail -3 $O/ventana.log | tr '\n' ' ')"; exit 1; }
log "ventana abierta; espero /health"

declare -A URL=([2x]=http://spark2:8000 [1x]=http://spark3:8000) DEP=([2x]=ursu-2x-head [1x]=ursu-1x) PID=()
t0=$(date +%s); pend="2x 1x"
while [ -n "$pend" ] && [ $(( $(date +%s) - t0 )) -lt 3600 ]; do
  for a in $pend; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 ${URL[$a]}/health)" = 200 ]; then
      log "$a LISTO en $(( ($(date +%s) - t0) / 60 )) min; arranca la batería"
      kubectl logs -n llm deploy/${DEP[$a]} > $O/arranque-$a.log 2>&1
      $D/chain.sh $a > $D/chain-$a.log 2>&1 & PID[$a]=$!
      pend=${pend/$a/}; pend=$(echo $pend)
    fi
  done
  for p in $(kubectl get pods -n llm -l dgx-847=prueba --no-headers 2>/dev/null | awk '$4>0 || $3 ~ /Error|CrashLoop|OOM/ {print $1"="$3"/"$4}'); do
    grep -qx "$p" /tmp/dgx847-vistos 2>/dev/null || { echo "$p" >> /tmp/dgx847-vistos; log "AVISO pod $p"; }
  done
  [ -n "$pend" ] && sleep 20
done
[ -n "$pend" ] && log "sin /health tras 60 min: $pend (logs en ops/arranque-*.log)" && for a in $pend; do kubectl logs -n llm deploy/${DEP[$a]} --tail=400 > $O/arranque-$a.log 2>&1; done

while [ "$(date +%s)" -lt $CIERRE ]; do
  vivos=0; for a in "${!PID[@]}"; do kill -0 ${PID[$a]} 2>/dev/null && vivos=1; done
  [ $vivos = 0 ] && break; sleep 30
done
for a in "${!PID[@]}"; do kill -0 ${PID[$a]} 2>/dev/null && { log "$a: corto la batería por la hora de cierre"; pkill -TERM -P ${PID[$a]}; kill -TERM ${PID[$a]}; }; done
sleep 30
for a in "${!PID[@]}"; do log "$a: $(tail -1 $D/chain-$a.log)"; done
log "cierre de la ventana"
$O/cierre.sh > $O/cierre.log 2>&1; log "cierre exit=$? ($(tail -1 $O/cierre.log))"
