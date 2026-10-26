#!/usr/bin/env bash
# DGX-932 vía 1, paso 2 (Dani 15-10): recalibra en dgx3, sin modelo, el prefill GDN en BF16 del TP=2 a partir de su volcado.
#  - El TP=1 de producción baja a 0 mientras dura (sirve el TP=2) y vuelve al final pase lo que pase (trap).
#  - Árbitro `sync-paused` con firma propia y auto-sync de root y ai apagado, como cadena-932.sh.
#  - Pods con la imagen del TP=2 (k94) y el parche B2, caché propia en dgx3 (/srv/ursu-cache/2x-b2-calib) con el fichero
#    de selecciones del TP=2: primero `--comprobar` (la clave casa con la del servidor), luego el ajuste.
#  - Al acabar bien, el fichero de selecciones va a /srv/ursu-cache/2x-b2 de dgx2 y dgx1 (la caché de B2; nunca a 2x).
set -u
export KUBECONFIG=~/.kube/config
W=/home/user/bench-results/dgx-932/calib; V=/home/user/bench-results/dgx-932/volcado
TP1=qwen38-flash-next-worker-dgx3; FIRMA=dgx-932-calib
IMG=ghcr.io/ursuciprian/spark-vllm-b12x@sha256:1346035488a5c8a0f405ade8e74febf68c66b8df1e026b7e3a6fb83cb9f4ce13
SEL=b12x/compile/preparation/225bf66521da31897db0719256f6a56cc38d43f87dd4f85f1f52549540986fe2.json
D=/srv/ursu-cache/2x-b2-calib; TOPE_S=${TOPE_S:-36000}; COMPILADORES=${COMPILADORES:-16}
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
estado() { kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'; }
escribe_estado() {
  python3 -c 'import json,sys; print(json.dumps({"data":{"compute_mode":json.dumps(json.load(sys.stdin),separators=(",",":"))}}))' \
    | kubectl patch cm -n comfyui gpu-arbiter-state --type merge --patch-file /dev/stdin >/dev/null
}
pausa() {
  estado > $W/arbitro-antes.json
  python3 - $W/arbitro-antes.json $FIRMA <<'PY' | escribe_estado
import json,sys,time
st=json.load(open(sys.argv[1]))
assert st.get("phase")=="ready" and st.get("desired_mode")==st.get("effective_mode")=="llm-tp", "árbitro no está en ready: %s" % st.get("phase")
print(json.dumps({**st,"phase":"sync-paused","sync_pause":{"by":sys.argv[2],"at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}}))
PY
}
reanuda() {
  estado | python3 -c '
import json,sys
st=json.load(sys.stdin)
if st.get("phase")=="sync-paused" and (st.get("sync_pause") or {}).get("by")==sys.argv[1]:
    print(json.dumps({**{k:v for k,v in st.items() if k!="sync_pause"},"phase":"ready","blockers":[],"last_error":None}))
else:
    sys.exit("árbitro en phase=%s sync_pause=%s: no es mi pausa, no lo toco" % (st.get("phase"), st.get("sync_pause")))' $FIRMA | escribe_estado
}
autosync_off() {
  for a in root ai; do kubectl get application -n argocd $a -o jsonpath='{.spec.syncPolicy}' > $W/$a-syncpolicy-antes.json; done
  for a in root ai; do kubectl patch application -n argocd $a --type json -p '[{"op":"remove","path":"/spec/syncPolicy/automated"}]' >/dev/null; done
}
autosync_on() {
  for a in root ai; do [ -s $W/$a-syncpolicy-antes.json ] && kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $W/$a-syncpolicy-antes.json)}}" >/dev/null; done
}
pod() {  # $1 nombre, $2 argumentos extra de recalibra_b12x.py: placement del TP=1 (runtime, toleration, nodo), GPU y caché propia
  kubectl get deploy -n llm $TP1 -o json > $W/tp1.json
  python3 - $W/tp1.json "$1" "$IMG" "$D" "$SEL" "$COMPILADORES" "${2:-}" <<'PY' | kubectl apply -f - >/dev/null
import json,sys
fichero,nombre,img,d,sel,comp,extra=sys.argv[1:]
tp1=json.load(open(fichero))["spec"]["template"]["spec"]
cmd=("QWEN_GDN_MIX=B2 python3 -I /cache/xdg/patch_gdn_mix.py && exec python3 -I /cache/xdg/recalibra_b12x.py "
     f"/cache/xdg/consultas.jsonl --seleccion /cache/xdg/{sel} --compiladores {comp} {extra}")
print(json.dumps({"apiVersion":"v1","kind":"Pod",
  "metadata":{"name":nombre,"namespace":"llm","labels":{"app":"calib-b2-tp2","dgx-932":"calib"}},
  "spec":{"restartPolicy":"Never","runtimeClassName":tp1["runtimeClassName"],"tolerations":tp1["tolerations"],
          "nodeSelector":tp1["nodeSelector"],
          "containers":[{"name":"calib","image":img,"command":["bash","-c",cmd],
                         "envFrom":[{"configMapRef":{"name":"qwen38fn-b12x-env-2x"}}],
                         "env":[{"name":"XDG_CACHE_HOME","value":"/cache/xdg"},{"name":"B12X_AUTOTUNE","value":"1"},
                                {"name":"PYTHONUNBUFFERED","value":"1"}],
                         "resources":{"limits":{"nvidia.com/gpu":"1","memory":"100Gi"},"requests":{"memory":"8Gi"}},
                         "volumeMounts":[{"name":"xdg","mountPath":"/cache/xdg"},{"name":"shm","mountPath":"/dev/shm"}]}],
          "volumes":[{"name":"xdg","hostPath":{"path":d,"type":"Directory"}},
                     {"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"8Gi"}}]}}))
PY
}
espera_pod() {  # $1 pod, $2 tope s → 0 si Succeeded; deja el log en $W/$1.log
  local t0=$(date +%s) ult=0 fase
  while :; do
    fase=$(kubectl get pod -n llm $1 -o jsonpath='{.status.phase}' 2>/dev/null)
    case "$fase" in Succeeded|Failed) kubectl logs -n llm $1 > $W/$1.log 2>&1; log "$1: $fase"; [ "$fase" = Succeeded ]; return; esac
    [ $(( $(date +%s)-t0 )) -gt $2 ] && { kubectl logs -n llm $1 > $W/$1.log 2>&1; log "FALLO: $1 sin terminar en $(( $2/60 )) min"; return 1; }
    [ $(( $(date +%s)-ult )) -ge 600 ] && { ult=$(date +%s); log "  $1 ($(( (ult-t0)/60 )) min): $(kubectl logs -n llm $1 --tail=200 2>/dev/null | grep -E 'PROGRESO|ajustando|consultas|caché' | tail -1 | cut -c1-200)"; }
    sleep 20
  done
}
vuelta() {
  trap - EXIT
  log "VUELTA: inicio"
  kubectl delete pod -n llm -l app=calib-b2-tp2 --ignore-not-found --wait=true >/dev/null
  kubectl scale deploy -n llm $TP1 --replicas=1 >/dev/null
  local t0=$(date +%s)
  until [ "$(kubectl get deploy -n llm $TP1 -o jsonpath='{.status.readyReplicas}')" = 1 ]; do
    [ $(( $(date +%s)-t0 )) -gt 2400 ] && { log "VUELTA: el TP=1 no está listo en 40 min; dejo el árbitro en pausa y el auto-sync apagado"; return 1; }
    sleep 15
  done
  log "VUELTA: TP=1 de producción listo en $(( ($(date +%s)-t0)/60 )) min"
  reanuda && log "VUELTA: árbitro en ready" || log "VUELTA: árbitro no reanudado (ver arriba)"
  autosync_on && log "VUELTA: auto-sync de root y ai restaurado"
  log "HECHO"
}

log "DGX-932 calibración en dgx3: inicio"
[ -s $V/consultas-rango0.jsonl ] || { log "FALLO: falta el volcado $V/consultas-rango0.jsonl"; exit 1; }
[ "$(kubectl get pods -n llm -l app=calib-b2-tp2 --no-headers 2>/dev/null)" = "" ] || { log "FALLO: ya hay pods calib-b2-tp2"; exit 1; }
ssh dgx3 "sudo -n mkdir -p $D/$(dirname $SEL)" || exit 1
for f in $W/recalibra_b12x.py:recalibra_b12x.py $V/consultas-rango0.jsonl:consultas.jsonl; do ssh dgx3 "sudo -n tee $D/${f#*:} >/dev/null" < ${f%%:*} || exit 1; done
kubectl get cm -n llm qwen38fn-b12x-recipe -o jsonpath='{.data.patch_gdn_mix\.py}' | ssh dgx3 "sudo -n tee $D/patch_gdn_mix.py >/dev/null" || exit 1
ssh dgx2 "sudo -n cat /srv/ursu-cache/2x-b2/$SEL" | ssh dgx3 "sudo -n tee $D/$SEL >/dev/null" || exit 1
ANTES=$(ssh dgx3 "sudo -n cat $D/$SEL" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["records"]))')
log "dgx3 preparado: script, volcado, parcheador y selecciones del TP=2 ($ANTES registros) en $D"
pausa || { log "FALLO: no pude pausar el árbitro"; exit 1; }
trap vuelta EXIT
log "árbitro en sync-paused ($FIRMA)"
autosync_off; log "auto-sync de root y ai apagado"
kubectl scale deploy -n llm $TP1 --replicas=0 >/dev/null
while kubectl get pods -n llm -l app=$TP1 --no-headers 2>/dev/null | grep -q .; do sleep 5; done
log "TP=1 de producción a 0"
pod calib-b2-comprobar --comprobar; espera_pod calib-b2-comprobar 1200 || { tail -5 $W/calib-b2-comprobar.log; exit 1; }
grep -E 'declaraciones|en caché|COMPROBADO' $W/calib-b2-comprobar.log | while read -r l; do log "  $l"; done
grep -q COMPROBADO $W/calib-b2-comprobar.log || exit 1
kubectl delete pod -n llm calib-b2-comprobar --wait=true >/dev/null
pod calib-b2-ajuste; log "ajuste en marcha (tope $(( TOPE_S/3600 )) h, $COMPILADORES compiladores)"
espera_pod calib-b2-ajuste $TOPE_S; ok=$?
grep -E '^m[0-9]+:|HECHO|INCOMPLETO' $W/calib-b2-ajuste.log | while read -r l; do log "  $l"; done
[ $ok = 0 ] && grep -q '^HECHO' $W/calib-b2-ajuste.log || { tail -20 $W/calib-b2-ajuste.log; exit 1; }
for n in dgx2 dgx; do ssh dgx3 "sudo -n cat $D/$SEL" | ssh $n "sudo -n tee /srv/ursu-cache/2x-b2/$SEL >/dev/null" || { log "FALLO: copia a $n"; exit 1; }; done
ssh dgx3 "sudo -n cat $D/$SEL" > $W/seleccion-b2.json
log "selecciones B2 en /srv/ursu-cache/2x-b2 de dgx2 y dgx1: $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' $W/seleccion-b2.json) registros (antes $ANTES)"
exit 0
