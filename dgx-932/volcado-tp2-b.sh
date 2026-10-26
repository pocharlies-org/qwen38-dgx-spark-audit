#!/usr/bin/env bash
# DGX-932 vía 1, paso 1, segundo intento (15-10): volcado de las consultas b12x del TP=2 en B2 sin autoajuste.
# El primero lo pisó la sync de DGX-940, que seguía reintentando (el head pasó su progressDeadline): ArgoCD reaplicó head y
# worker encima del cambio a B2. Ahora producción no se modifica: su head y su worker bajan a 0 y el arranque B2 va en dos
# Deployments de prueba (b2v-head, b2v-worker) que ArgoCD no sigue. Con producción a 0 se quitan además las dos variables que
# dejó el primer intento (B2_AUTOTUNE, VLLM_B12X_DUMP_QUERIES). Al acabar, producción vuelve a 1 tal cual está en git.
#  - Solo empieza con la operación de ArgoCD de `ai` terminada, la app Synced y Healthy, el árbitro en ready y el TP=2 listo.
#  - Árbitro `sync-paused` con firma propia, auto-sync de root y ai apagado y la voz a 0 (como el hook llm-deploy-pause).
#  - Caché aparte (/srv/ursu-cache/2x-b2, copia de 2x en dgx2 y dgx1).
set -u
export KUBECONFIG=~/.kube/config
W=/home/user/bench-results/dgx-932/volcado; mkdir -p $W
FIRMA=dgx-932-volcado; HEAD=qwen38-flash-next-head; WORKER=qwen38-flash-next-worker; R2=http://spark2:8000
VOZ="omnivoice-tts-nvfp4 stt-turbo"; NODOS="dgx2 dgx"   # ssh: dgx2 = head (rango 0), dgx = worker (rango 1)
DUMP=/cache/xdg/b12x-consultas.jsonl; DUMP_HOST=/srv/ursu-cache/2x-b2/b12x-consultas.jsonl
SOBRAN="B2_AUTOTUNE VLLM_B12X_DUMP_QUERIES"
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
estado() { kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'; }
fase() { estado | python3 -c 'import json,sys; print(json.load(sys.stdin).get("phase"))'; }
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
argo_quieto() {  # sin operación en marcha (un reintento pendiente sigue en Running), Synced y Healthy
  case "$(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase}|{.status.sync.status}|{.status.health.status}')" in
    Succeeded"|Synced|Healthy"|Failed"|Synced|Healthy"|Error"|Synced|Healthy") return 0 ;; *) return 1 ;;
  esac
}
listo_head() {
  [ "$(kubectl get deploy -n llm $HEAD -o jsonpath='{.status.readyReplicas}')" = 1 ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R2/health)" = 200 ]
}
sin_sobrantes() {  # quita de head y worker de producción las variables del primer intento (con réplicas a 0: no rearranca nada)
  for d in $HEAD $WORKER; do
    local p; p=$(kubectl get deploy -n llm $d -o json | python3 -c '
import json,sys
c=json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0]
print(json.dumps([{"op":"remove","path":"/spec/template/spec/containers/0/env/%d" % i} for i,e in reversed(list(enumerate(c["env"]))) if e["name"] in sys.argv[1].split()]))' "$SOBRAN")
    [ "$p" = "[]" ] || kubectl patch deploy -n llm $d --type json -p "$p" >/dev/null || return 1
  done
}
prueba() {  # $1 deploy de producción guardado → Deployment de prueba en B2 con el volcado, sin rastro de ArgoCD
  python3 - "$1" "$DUMP" "$SOBRAN" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); dump=sys.argv[2]; sobran=sys.argv[3].split()
nombre={"qwen38-flash-next-head":"b2v-head","qwen38-flash-next-worker":"b2v-worker"}[d["metadata"]["name"]]
t=d["spec"]["template"]; s=t["spec"]; c=s["containers"][0]
env={e["name"]:e for e in c["env"] if e["name"] not in sobran}
assert env["QWEN_GDN_MIX"]["value"]=="A", env["QWEN_GDN_MIX"]
env["QWEN_GDN_MIX"]={"name":"QWEN_GDN_MIX","value":"B2"}
env["B2_AUTOTUNE"]={"name":"B2_AUTOTUNE","value":"0"}
env["VLLM_B12X_DUMP_QUERIES"]={"name":"VLLM_B12X_DUMP_QUERIES","value":dump}
c["env"]=list(env.values())
xdg=[v for v in s["volumes"] if v["name"]=="xdg"]
assert len(xdg)==1 and xdg[0]["hostPath"]["path"]=="/srv/ursu-cache/2x", xdg
xdg[0]["hostPath"]["path"]="/srv/ursu-cache/2x-b2"
if nombre=="b2v-worker":  # sin worker-sigue-head: busca el head de producción, que está a 0
    c["args"]=["bash","/opt/ursu-mod/run.sh","/etc/ursu/serve-gdn-mix.sh"]
    c.pop("livenessProbe",None)
lab={"app":nombre,"dgx-932":"volcado"}
t["metadata"]={"labels":lab,"annotations":{k:v for k,v in (t["metadata"].get("annotations") or {}).items() if not k.startswith("argocd")}}
print(json.dumps({"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":nombre,"namespace":"llm","labels":lab},
  "spec":{"replicas":1,"strategy":{"type":"Recreate"},"selector":{"matchLabels":{"app":nombre}},"template":t}}))
PY
}
volcado_hecho() {
  for n in $NODOS; do
    local a b; a=$(ssh $n "sudo -n stat -c %s $DUMP_HOST 2>/dev/null"); [ -n "$a" ] || return 1
    ssh $n "sudo -n grep -q '\"stage\": \"state\"' $DUMP_HOST" || return 1
    sleep 20; b=$(ssh $n "sudo -n stat -c %s $DUMP_HOST"); [ "$a" = "$b" ] || return 1
  done
}
baja_pruebas() {
  kubectl delete deploy -n llm b2v-head b2v-worker --ignore-not-found --wait=true >/dev/null
  while kubectl get pods -n llm -l dgx-932=volcado --no-headers 2>/dev/null | grep -q .; do sleep 5; done
}
vuelta() {
  trap - EXIT
  log "VUELTA: inicio"
  for d in b2v-head b2v-worker; do kubectl logs -n llm deploy/$d --tail=-1 > $W/$d.log 2>&1; done
  baja_pruebas; log "VUELTA: pruebas fuera"
  sin_sobrantes || log "VUELTA: no pude quitar las variables sobrantes (ver arriba)"
  kubectl scale deploy -n llm $HEAD $WORKER --replicas=1 >/dev/null
  local t0=$(date +%s)
  until listo_head; do
    [ $(( $(date +%s)-t0 )) -gt 3600 ] && { log "VUELTA: el TP=2 no está listo en 60 min; dejo árbitro en pausa, voz a 0 y auto-sync apagado"; return 1; }
    sleep 20
  done
  log "VUELTA: TP=2 de producción listo en $(( ($(date +%s)-t0)/60 )) min"
  for d in $VOZ; do kubectl scale deploy -n llm $d --replicas=$(cat $W/$d.replicas) >/dev/null; done; log "VUELTA: voz a como estaba"
  reanuda && log "VUELTA: árbitro en ready" || log "VUELTA: árbitro no reanudado (ver arriba)"
  autosync_on && log "VUELTA: auto-sync de root y ai restaurado"
  sleep 120; log "VUELTA: ArgoCD ai $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status} {.status.health.status} op={.status.operationState.phase}')"
  log "HECHO"
}

log "DGX-932 volcado (2): espero a ArgoCD quieto, árbitro en ready y TP=2 listo"
t0=$(date +%s)
until argo_quieto && [ "$(fase)" = ready ] && listo_head; do
  [ $(( $(date +%s)-t0 )) -gt 10800 ] && { log "FALLO: sin las condiciones de partida en 3 h ($(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase}|{.status.sync.status}|{.status.health.status}'), árbitro $(fase))"; exit 1; }
  sleep 30
done
[ "$(kubectl get pods -n llm -l dgx-932=volcado --no-headers 2>/dev/null)" = "" ] || { log "FALLO: ya hay pods de prueba"; exit 1; }
for d in $HEAD $WORKER; do kubectl get deploy -n llm $d -o json > $W/$d-antes.json; done
for d in $VOZ; do kubectl get deploy -n llm $d -o jsonpath='{.spec.replicas}' > $W/$d.replicas; done
pausa || { log "FALLO: no pude pausar el árbitro"; exit 1; }
trap vuelta EXIT
log "árbitro en sync-paused ($FIRMA)"
autosync_off; log "auto-sync de root y ai apagado"
for d in $VOZ; do kubectl scale deploy -n llm $d --replicas=0 >/dev/null; done
kubectl scale deploy -n llm $HEAD $WORKER --replicas=0 >/dev/null
until [ -z "$(kubectl get pods -n llm --no-headers 2>/dev/null | grep -E '^(omnivoice-tts-nvfp4|stt-turbo|qwen38-flash-next-head|qwen38-flash-next-worker)-[0-9a-f]+-[0-9a-z]+ ')" ]; do sleep 5; done
log "voz y TP=2 de producción a 0"
sin_sobrantes || { log "FALLO: no pude quitar las variables sobrantes"; exit 1; }
log "producción sin $SOBRAN"
for n in $NODOS; do ssh $n 'sudo -n rm -rf /srv/ursu-cache/2x-b2 && sudo -n cp -a /srv/ursu-cache/2x /srv/ursu-cache/2x-b2 && sudo -n rm -f /srv/ursu-cache/2x-b2/b12x-consultas.jsonl' || { log "FALLO: copia de la caché en $n"; exit 1; }; done
log "caché copiada a /srv/ursu-cache/2x-b2 en dgx2 y dgx1"
for d in $HEAD $WORKER; do prueba $W/$d-antes.json | kubectl apply -f - >/dev/null || { log "FALLO: no pude crear la prueba de $d"; exit 1; }; done
sleep 60
for p in b2v-head b2v-worker; do
  e=$(kubectl get pods -n llm -l app=$p -o jsonpath='{.items[0].spec.containers[0].env[?(@.name=="QWEN_GDN_MIX")].value} {.items[0].spec.volumes[?(@.name=="xdg")].hostPath.path}')
  [ "$e" = "B2 /srv/ursu-cache/2x-b2" ] || { log "FALLO: $p no arrancó en B2 con 2x-b2 ($e)"; exit 1; }
done
log "B2 sin autoajuste: b2v-head y b2v-worker arrancando con el volcado en $DUMP"
t0=$(date +%s); ult=0
until volcado_hecho; do
  rs=$(kubectl get pods -n llm -l app=b2v-head -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' 2>/dev/null)
  [ "${rs:-0}" = 0 ] || { log "FALLO: b2v-head se reinició ($rs)"; exit 1; }
  [ $(( $(date +%s)-t0 )) -gt 2700 ] && { log "FALLO: sin volcado completo en 45 min"; exit 1; }
  [ $(( $(date +%s)-ult )) -ge 300 ] && { ult=$(date +%s); log "  esperando ($(( (ult-t0)/60 )) min): $(kubectl logs -n llm deploy/b2v-head --tail=1 2>/dev/null | cut -c1-150)"; }
  sleep 15
done
log "volcado completo en $(( ($(date +%s)-t0)/60 )) min"
ssh dgx2 "sudo -n cat $DUMP_HOST" > $W/consultas-rango0.jsonl; ssh dgx "sudo -n cat $DUMP_HOST" > $W/consultas-rango1.jsonl
log "consultas: rango0 $(wc -l < $W/consultas-rango0.jsonl) líneas, rango1 $(wc -l < $W/consultas-rango1.jsonl); $(python3 -c '
import json,sys,collections
c=collections.Counter((r["stage"],r["component"]) for r in map(json.loads,open(sys.argv[1])))
print(", ".join("%s/%s=%d"%(k[0],k[1],v) for k,v in sorted(c.items()) if k[0]=="state"))' $W/consultas-rango0.jsonl)"
exit 0
