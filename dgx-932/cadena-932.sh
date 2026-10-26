#!/usr/bin/env bash
# DGX-932 (Dani 15-10): B2 (GDN b12x en prefill y decode, estado BF16, registros compactos + checkpoints diferidos) con
# autoajuste BF16 en dgx3, medido contra A (decode b12x + prefill triton) en el MISMO pod de pruebas y con la misma imagen.
# El TP=2 de producción no se toca. El TP=1 de producción baja a 0 mientras dura y vuelve al final pase lo que pase (trap).
#  - Árbitro: `sync-paused` firmado `dgx-932-b2` (como el hook llm-deploy-pause, SIN tocar la voz): con la fase en ready su
#    reparación de deriva volvería a subir el TP=1 y su vigilante podría reiniciar el perfil entero a los 30 min.
#  - Auto-sync de root y ai apagado mientras dura (como ventana-918b.sh): una sync de ai correría los hooks de pausa/vuelta.
#  - Pod de pruebas `b2-dgx3` con etiquetas propias (el Service del TP=1 y LiteLLM no lo ven) y caché aparte (1x-b2).
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh   # log, D, O, TOOLS
export KUBECONFIG=~/.kube/config
W=/home/user/bench-results/dgx-932; R1=http://spark3:8000; export MODEL=qwen3.8-flash-next
TP1=qwen38-flash-next-worker-dgx3; FIRMA=dgx-932-b2
B2_TOPE_S=${B2_TOPE_S:-32400}   # 9 h para el autoajuste + arranque

estado() { kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'; }
escribe_estado() {  # stdin: JSON del estado nuevo (solo la clave compute_mode; las demás del CM no se tocan)
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
reanuda() {  # solo deshace MI pausa
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
manifiesto() {  # $1 = A|B2: el TP=1 de producción con otro nombre, etiquetas propias, caché aparte y arranque de hasta 9 h
  python3 - $W/tp1-prod.json $1 <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); modo=sys.argv[2]
lab={"app":"b2-dgx3","dgx-932":"b2"}
pod=d["spec"]["template"]["spec"]
c=pod["containers"][0]
env={e["name"]:e for e in c["env"]}
env["QWEN_GDN_MIX"]={"name":"QWEN_GDN_MIX","value":modo}
if modo=="B2": env["B2_AUTOTUNE"]={"name":"B2_AUTOTUNE","value":"1"}
c["env"]=list(env.values())
c["startupProbe"]["failureThreshold"]=1080   # 30 s x 1080 = 9 h
for v in pod["volumes"]:
    if (v.get("hostPath") or {}).get("path")=="/srv/ursu-cache/1x": v["hostPath"]["path"]="/srv/ursu-cache/1x-b2"
assert any((v.get("hostPath") or {}).get("path")=="/srv/ursu-cache/1x-b2" for v in pod["volumes"]), "sin caché aparte"
print(json.dumps({"apiVersion":"apps/v1","kind":"Deployment",
  "metadata":{"name":"b2-dgx3","namespace":"llm","labels":lab},
  "spec":{"replicas":1,"strategy":{"type":"Recreate"},"selector":{"matchLabels":{"app":"b2-dgx3"}},
          "template":{"metadata":{"labels":lab},"spec":pod}}}))
PY
}
espera_listo() {  # $1 tope en s: /health 200 del pod de pruebas; un reinicio del contenedor = fallo
  local t0=$(date +%s) ult=0
  while :; do
    local rs; rs=$(kubectl get pods -n llm -l app=b2-dgx3 -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}' 2>/dev/null)
    [ "${rs:-0}" -gt 0 ] && { log "FALLO: el contenedor de b2-dgx3 se reinició"; kubectl logs -n llm -l app=b2-dgx3 --previous --tail=40 > $W/fallo-previo.log 2>&1; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R1/health)" = 200 ] && { log "listo en $(( ($(date +%s)-t0)/60 )) min"; return 0; }
    [ $(( $(date +%s)-t0 )) -gt $1 ] && { log "FALLO: sin /health en $(( $1/60 )) min"; return 1; }
    [ $(( $(date +%s)-ult )) -ge 600 ] && { ult=$(date +%s); log "  esperando ($(( (ult-t0)/60 )) min): $(kubectl logs -n llm -l app=b2-dgx3 --tail=1 2>/dev/null | cut -c1-140)"; }
    sleep 20
  done
}
mide_todo() {  # $1 etiqueta (A|B2): tools, decode c1/c4/c8 y prefill, igual que cadena-tp21-A.sh y prefill.sh
  local T=$W/$1; mkdir -p $T
  kubectl logs -n llm -l app=b2-dgx3 --tail=-1 > $T/arranque.log 2>&1
  log "$1 KV: $(grep -hoE 'GPU KV cache size: [0-9,]+ tokens|Available KV cache memory: [0-9.]+ GiB|block_size[=: ]+[0-9]+' $T/arranque.log | sort -u | paste -sd' ' | cut -c1-160)"
  python3 $D/tools-check.py $R1 $T/tools > $T/tools.log 2>&1; log "$1 tools exit=$? $(tail -1 $T/tools.log | cut -c1-100)"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $R1 --model $MODEL --concurrency 1,4,8 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 \
    --long-tokens 32000 --max-model-len 262144 --seed 496368 --out $T/decode.json > $T/decode.log 2>&1
  log "$1 decode $(python3 $O/resumen-decode.py $T/decode.json)"
  bash $O/prefill.sh $R1 $T/prefill > $T/prefill.log 2>&1; log "$1 prefill $(tr '\n' ' ' < $T/prefill.log)"
}
baja_pruebas() {
  kubectl delete deploy -n llm b2-dgx3 --ignore-not-found --wait=true >/dev/null
  while kubectl get pods -n llm -l app=b2-dgx3 --no-headers 2>/dev/null | grep -q .; do sleep 5; done
}
vuelta() {  # siempre: pruebas fuera, TP=1 de producción arriba, árbitro en ready, auto-sync como estaba
  trap - EXIT
  log "VUELTA: inicio"
  baja_pruebas
  kubectl scale deploy -n llm $TP1 --replicas=1 >/dev/null
  local t0=$(date +%s)
  until [ "$(kubectl get deploy -n llm $TP1 -o jsonpath='{.status.readyReplicas}')" = 1 ]; do
    [ $(( $(date +%s)-t0 )) -gt 2400 ] && { log "VUELTA: el TP=1 no está listo en 40 min; dejo el árbitro en pausa y el auto-sync apagado para no disparar su reparación"; return 1; }
    sleep 15
  done
  log "VUELTA: TP=1 de producción listo en $(( ($(date +%s)-t0)/60 )) min"
  reanuda && log "VUELTA: árbitro en ready" || log "VUELTA: árbitro no reanudado (ver arriba)"
  autosync_on && log "VUELTA: auto-sync de root y ai restaurado"
  log "HECHO"
}

log "DGX-932: inicio"
[ "$(kubectl get deploy -n llm b2-dgx3 --ignore-not-found -o name)" = "" ] || { log "FALLO: ya existe b2-dgx3"; exit 1; }
pausa || { log "FALLO: no pude pausar el árbitro"; exit 1; }
trap vuelta EXIT
log "árbitro en sync-paused ($FIRMA)"
autosync_off; log "auto-sync de root y ai apagado"
kubectl scale deploy -n llm $TP1 --replicas=0 >/dev/null
while kubectl get pods -n llm -l app=$TP1 --no-headers 2>/dev/null | grep -q .; do sleep 5; done
log "TP=1 de producción a 0"
ssh dgx3 'sudo -n rm -rf /srv/ursu-cache/1x-b2 && sudo -n cp -a /srv/ursu-cache/1x /srv/ursu-cache/1x-b2' || { log "FALLO: copia de la caché"; exit 1; }
log "caché copiada a /srv/ursu-cache/1x-b2 ($(ssh dgx3 'sudo -n du -sh /srv/ursu-cache/1x-b2' | cut -f1))"

manifiesto A | kubectl apply -f - >/dev/null; log "A: arrancando b2-dgx3"
espera_listo 2700 || exit 1
mide_todo A
baja_pruebas

manifiesto B2 | kubectl apply -f - >/dev/null; log "B2: arrancando b2-dgx3 con autoajuste (tope $(( B2_TOPE_S/3600 )) h)"
espera_listo $B2_TOPE_S || exit 1
log "B2 LISTO"
mide_todo B2
exit 0
