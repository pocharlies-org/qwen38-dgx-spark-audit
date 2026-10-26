#!/usr/bin/env bash
# DGX-932 vía 1, paso 1 (Dani 15-10): arranque corto del TP=2 en B2 sin autoajuste para volcar sus consultas b12x
# (VLLM_B12X_DUMP_QUERIES). Se vuelve a A en cuanto los dos rangos han volcado la etapa `state`; el TP=1 atiende mientras.
#  - Espera a que DGX-940 deje el TP=2 con la KV nueva (launch-revision …dgx940) y el árbitro en ready.
#  - Árbitro `sync-paused` con firma propia, auto-sync de root y ai apagado y la voz a 0, como el hook llm-deploy-pause.
#  - Caché aparte (/srv/ursu-cache/2x-b2, copia de 2x en dgx1 y dgx2): la de producción no se toca.
#  - Vuelta (trap): head y worker exactamente como estaban (kubectl replace del objeto guardado), voz, árbitro y auto-sync.
set -u
export KUBECONFIG=~/.kube/config
W=/home/user/bench-results/dgx-932/volcado; mkdir -p $W
FIRMA=dgx-932-volcado; HEAD=qwen38-flash-next-head; WORKER=qwen38-flash-next-worker; R2=http://spark2:8000
VOZ="omnivoice-tts-nvfp4 stt-turbo"; NODOS="dgx2 dgx"   # ssh: dgx2 = head (rango 0), dgx = worker (rango 1)
DUMP=/cache/xdg/b12x-consultas.jsonl; DUMP_HOST=/srv/ursu-cache/2x-b2/b12x-consultas.jsonl
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
limpio() {  # $1 deploy guardado → objeto para kubectl replace (sin estado ni campos del servidor); $2 = B2 para el modo volcado
  python3 - "$1" "${2:-}" "$DUMP" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); modo=sys.argv[2]
for k in ("resourceVersion","uid","creationTimestamp","generation","managedFields"): d["metadata"].pop(k,None)
d.pop("status",None)
if modo=="B2":
    t=d["spec"]["template"]; c=t["spec"]["containers"][0]
    env={e["name"]:e for e in c["env"]}
    assert env["QWEN_GDN_MIX"]["value"]=="A"
    env["QWEN_GDN_MIX"]={"name":"QWEN_GDN_MIX","value":"B2"}
    env["B2_AUTOTUNE"]={"name":"B2_AUTOTUNE","value":"0"}
    env["VLLM_B12X_DUMP_QUERIES"]={"name":"VLLM_B12X_DUMP_QUERIES","value":sys.argv[3]}
    c["env"]=list(env.values())
    xdg=[v for v in t["spec"]["volumes"] if v["name"]=="xdg"]
    assert len(xdg)==1 and xdg[0]["hostPath"]["path"]=="/srv/ursu-cache/2x", xdg
    xdg[0]["hostPath"]["path"]="/srv/ursu-cache/2x-b2"
    a=t["metadata"]["annotations"]; a["internal.example/launch-revision"]+="-b2volcado"
print(json.dumps(d))
PY
}
listo_head() {  # pod del head con la revisión $1, Ready y /health 200
  kubectl get pods -n llm -l app=$HEAD -o json | python3 -c '
import json,sys
for p in json.load(sys.stdin)["items"]:
    a=p["metadata"].get("annotations",{}).get("internal.example/launch-revision","")
    if a==sys.argv[1] and p["metadata"].get("deletionTimestamp") is None and all(c.get("ready") for c in p["status"].get("containerStatuses",[{}])): sys.exit(0)
sys.exit(1)' "$1" && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R2/health)" = 200 ]
}
volcado_hecho() {  # los dos rangos han escrito la etapa state y el fichero no crece en 20 s
  for n in $NODOS; do
    local a b; a=$(ssh $n "sudo -n stat -c %s $DUMP_HOST 2>/dev/null"); [ -n "$a" ] || return 1
    ssh $n "sudo -n grep -q '\"stage\": \"state\"' $DUMP_HOST" || return 1
    sleep 20; b=$(ssh $n "sudo -n stat -c %s $DUMP_HOST"); [ "$a" = "$b" ] || return 1
  done
}
vuelta() {
  trap - EXIT
  log "VUELTA: inicio"
  for d in $HEAD $WORKER; do kubectl logs -n llm deploy/$d --tail=-1 > $W/$d-b2.log 2>&1; done
  for d in $HEAD $WORKER; do limpio $W/$d-antes.json | kubectl replace -f - >/dev/null && log "VUELTA: $d como estaba"; done
  local t0=$(date +%s) rev; rev=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["spec"]["template"]["metadata"]["annotations"]["internal.example/launch-revision"])' $W/$HEAD-antes.json)
  until listo_head "$rev"; do
    [ $(( $(date +%s)-t0 )) -gt 2700 ] && { log "VUELTA: el TP=2 no está listo en 45 min; dejo árbitro en pausa, voz a 0 y auto-sync apagado"; return 1; }
    sleep 20
  done
  log "VUELTA: TP=2 en A listo en $(( ($(date +%s)-t0)/60 )) min"
  for d in $VOZ; do kubectl scale deploy -n llm $d --replicas=$(cat $W/$d.replicas) >/dev/null; done; log "VUELTA: voz a como estaba"
  reanuda && log "VUELTA: árbitro en ready" || log "VUELTA: árbitro no reanudado (ver arriba)"
  autosync_on && log "VUELTA: auto-sync de root y ai restaurado"
  sleep 90; log "VUELTA: ArgoCD ai $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status} {.status.health.status}')"
  log "HECHO"
}

log "DGX-932 volcado: espero a DGX-940 (TP=2 con la KV nueva) y al árbitro en ready"
t0=$(date +%s)
until listo_head "ursu-b12x-2mas1-20261015-dgx940" && [ "$(estado | python3 -c 'import json,sys; s=json.load(sys.stdin); print(s.get("phase"))')" = ready ]; do
  [ $(( $(date +%s)-t0 )) -gt 7200 ] && { log "FALLO: DGX-940 no ha dejado el TP=2 listo en 2 h"; exit 1; }
  sleep 30
done
[ "$(kubectl get deploy -n llm stt-turbo -o jsonpath='{.spec.replicas}')" -ge 1 ] || { log "FALLO: la voz no ha vuelto tras DGX-940"; exit 1; }
for d in $HEAD $WORKER; do kubectl get deploy -n llm $d -o json > $W/$d-antes.json; done
for d in $VOZ; do kubectl get deploy -n llm $d -o jsonpath='{.spec.replicas}' > $W/$d.replicas; done
pausa || { log "FALLO: no pude pausar el árbitro"; exit 1; }
trap vuelta EXIT
log "árbitro en sync-paused ($FIRMA)"
autosync_off; log "auto-sync de root y ai apagado"
for d in $VOZ; do kubectl scale deploy -n llm $d --replicas=0 >/dev/null; done
until [ -z "$(kubectl get pods -n llm --no-headers 2>/dev/null | grep -E '^(omnivoice-tts-nvfp4|stt-turbo)-')" ]; do sleep 5; done
log "voz a 0"
for n in $NODOS; do ssh $n 'sudo -n rm -rf /srv/ursu-cache/2x-b2 && sudo -n cp -a /srv/ursu-cache/2x /srv/ursu-cache/2x-b2 && sudo -n rm -f /srv/ursu-cache/2x-b2/b12x-consultas.jsonl' || { log "FALLO: copia de la caché en $n"; exit 1; }; done
log "caché copiada a /srv/ursu-cache/2x-b2 en dgx2 y dgx1"
for d in $HEAD $WORKER; do limpio $W/$d-antes.json B2 | kubectl replace -f - >/dev/null || { log "FALLO: replace de $d en B2"; exit 1; }; done
log "B2 sin autoajuste: arrancando head y worker con el volcado en $DUMP"
t0=$(date +%s); ult=0
until volcado_hecho; do
  rs=$(kubectl get pods -n llm -l app=$HEAD -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' 2>/dev/null)
  case " $rs " in *" 0 "*|"  ") ;; *) log "FALLO: el head se reinició ($rs)"; exit 1 ;; esac
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && { log "FALLO: sin volcado completo en 30 min"; exit 1; }
  [ $(( $(date +%s)-ult )) -ge 180 ] && { ult=$(date +%s); log "  esperando ($(( (ult-t0)/60 )) min): $(kubectl logs -n llm deploy/$HEAD --tail=1 2>/dev/null | cut -c1-150)"; }
  sleep 15
done
log "volcado completo en $(( ($(date +%s)-t0)/60 )) min"
ssh dgx2 "sudo -n cat $DUMP_HOST" > $W/consultas-rango0.jsonl; ssh dgx "sudo -n cat $DUMP_HOST" > $W/consultas-rango1.jsonl
log "consultas: rango0 $(wc -l < $W/consultas-rango0.jsonl) líneas, rango1 $(wc -l < $W/consultas-rango1.jsonl); $(python3 -c '
import json,sys,collections
c=collections.Counter((r["stage"],r["component"]) for r in map(json.loads,open(sys.argv[1])))
print(", ".join("%s/%s=%d"%(k[0],k[1],v) for k,v in sorted(c.items())))' $W/consultas-rango0.jsonl)"
exit 0
