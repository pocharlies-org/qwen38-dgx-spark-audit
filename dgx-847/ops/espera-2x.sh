# DGX-847: reinicio y espera de la 2×. Uso: . espera-2x.sh; reinicia_2x [ARGS de kubectl set env]; espera_2x
# - Un lado que reinicia deja al otro colgado del rendezvous viejo (TCPStore: Broken pipe): se bajan y suben los dos juntos.
# - Al bajar, NVRM tarda en devolver la memoria unificada: subir antes colgó el driver de dgx1 (NV_ERR_NO_MEMORY, 23:24Z),
#   así que se espera a tener >=100 GB disponibles en dgx y dgx2 antes de subir.
SEL_2X='app in (ursu-2x-head,ursu-2x-worker)'
baja_2x() {
  kubectl scale -n llm deploy/ursu-2x-head deploy/ursu-2x-worker --replicas=0 >/dev/null
  while kubectl get pods -n llm -l "$SEL_2X" --no-headers 2>/dev/null | grep -q .; do sleep 5; done
  local h t0=$(date +%s)
  for h in dgx dgx2; do
    until [ "$(ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")" -ge 100 ]; do
      [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "$h no recupera la memoria en 15 min"; return 1; }; sleep 10
    done
  done
  log "2× abajo; memoria libre en dgx y dgx2"
}
reinicia_2x() {
  baja_2x || return 1
  [ $# -gt 0 ] && for d in ursu-2x-head ursu-2x-worker; do kubectl set env -n llm deploy/$d "$@" >/dev/null; done
  kubectl scale -n llm deploy/ursu-2x-head deploy/ursu-2x-worker --replicas=1 >/dev/null
}
espera_2x() {
  local t0=$(date +%s)
  sleep 15
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R/health)" = 200 ]; do
    if kubectl get pods -n llm -l "$SEL_2X" -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | grep -qE '[1-9]'; then
      log "un lado de la 2× reinició: los dos abajo y arriba juntos"; reinicia_2x || return 1
    fi
    sleep 15
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "sin /health en 60 min"; return 1; }
  done
  return 0  # el until devuelve el estado del último [ ... -gt 3600 ] (1), no el del /health
}
