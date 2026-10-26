#!/usr/bin/env bash
# DGX-932 vía 1, paso 3 (16-10): el TP=2 en B2 con las 10 selecciones BF16 calibradas en dgx3, medido frente a A en la misma
# ventana, con los mismos pods de prueba, la misma caché y la misma KV.
#  - Autoajuste ENCENDIDO en los dos: con B12X_AUTOTUNE=0 b12x no consulta la caché (session.py `_warmup_only`: config por
#    defecto en todo lo ajustable; el volcado dio 0 en caché frente a 456 de A). Con 1 consulta las 692 selecciones de 2x-b2
#    y solo mediría lo que faltara: "medidos máx 0" en el arranque = todo salió de la caché.
#  - KV de 25,5e9 bytes en los dos modos (EXTRA_ARGS: argparse se queda con el último valor): B2 pide ~5 GB más por nodo y
#    con la KV de producción dgx2 se quedaba en ~3 GB libres. La velocidad no depende del tamaño de la KV.
#  - Producción no se modifica: head y worker a 0 y los modos en Deployments de prueba (b2m-head, b2m-worker) que ArgoCD no
#    sigue, con etiquetas propias (ni el Service ni LiteLLM los ven). Árbitro sync-paused con firma propia, auto-sync de
#    root y ai apagado y la voz a 0, como el volcado. Producción vuelve a 1 tal cual está en git pase lo que pase (trap).
#  - Antes de cada arranque compacta la memoria de dgx2 y dgx1: con bloques grandes la KV se reservó en 7 min, sin ellos en 33.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh   # log, D, O, TOOLS
export KUBECONFIG=~/.kube/config
W=/home/user/bench-results/dgx-932/b2-tp2; mkdir -p $W
FIRMA=dgx-932-b2tp2; HEAD=qwen38-flash-next-head; WORKER=qwen38-flash-next-worker; R2=http://spark2:8000
R1=http://spark3:8000; export MODEL=qwen3.8-flash-next
VOZ="omnivoice-tts-nvfp4 stt-turbo"; NODOS="dgx2 dgx"   # ssh: dgx2 = head (rango 0), dgx = worker (rango 1)
KV_PRUEBA=25500000000; MODOS=${MODOS:-"A B2"}
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
sano() { [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $1/health)" = 200 ]; }
listo_head() { [ "$(kubectl get deploy -n llm $HEAD -o jsonpath='{.status.readyReplicas}')" = 1 ] && sano $R2; }
prueba() {  # $1 deploy de producción guardado, $2 modo → Deployment de prueba sin rastro de ArgoCD
  python3 - "$1" "$2" "$KV_PRUEBA" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); modo=sys.argv[2]; kv=sys.argv[3]
nombre={"qwen38-flash-next-head":"b2m-head","qwen38-flash-next-worker":"b2m-worker"}[d["metadata"]["name"]]
t=d["spec"]["template"]; s=t["spec"]; c=s["containers"][0]
env={e["name"]:e for e in c["env"]}
assert env["QWEN_GDN_MIX"]["value"]=="A" and not {"B2_AUTOTUNE","VLLM_B12X_DUMP_QUERIES","EXTRA_ARGS"} & set(env), sorted(env)
env["QWEN_GDN_MIX"]={"name":"QWEN_GDN_MIX","value":modo}
env["B2_AUTOTUNE"]={"name":"B2_AUTOTUNE","value":"1"}   # solo lo lee B2; A ya ajusta con la receta (B12X_AUTOTUNE=1)
env["EXTRA_ARGS"]={"name":"EXTRA_ARGS","value":"--kv-cache-memory-bytes "+kv}
c["env"]=list(env.values())
xdg=[v for v in s["volumes"] if v["name"]=="xdg"]
assert len(xdg)==1 and xdg[0]["hostPath"]["path"]=="/srv/ursu-cache/2x", xdg
xdg[0]["hostPath"]["path"]="/srv/ursu-cache/2x-b2"
if nombre=="b2m-worker":  # sin worker-sigue-head: busca el head de producción, que está a 0
    c["args"]=["bash","/opt/ursu-mod/run.sh","/etc/ursu/serve-gdn-mix.sh"]
    c.pop("livenessProbe",None)
lab={"app":nombre,"dgx-932":"medida"}
t["metadata"]={"labels":lab,"annotations":{k:v for k,v in (t["metadata"].get("annotations") or {}).items() if not k.startswith("argocd")}}
print(json.dumps({"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":nombre,"namespace":"llm","labels":lab},
  "spec":{"replicas":1,"strategy":{"type":"Recreate"},"selector":{"matchLabels":{"app":nombre}},"template":t}}))
PY
}
baja_pruebas() {
  kubectl delete deploy -n llm b2m-head b2m-worker --ignore-not-found --wait=true >/dev/null
  while kubectl get pods -n llm -l dgx-932=medida --no-headers 2>/dev/null | grep -q .; do sleep 5; done
}
compacta() {  # bloques libres de orden >= 9 (2 MiB+) antes y después de compactar
  for n in $NODOS; do
    local a b; a=$(ssh $n "awk '/Normal/{s=0; for(i=14;i<=NF;i++) s+=\$i; print s}' /proc/buddyinfo | paste -sd+ | bc")
    ssh $n "sudo -n sh -c 'echo 1 > /proc/sys/vm/compact_memory'"
    b=$(ssh $n "awk '/Normal/{s=0; for(i=14;i<=NF;i++) s+=\$i; print s}' /proc/buddyinfo | paste -sd+ | bc")
    log "  $n: bloques de 2 MiB+ libres $a → $b, MemAvailable $(ssh $n "awk '/MemAvailable/{printf \"%.1f GB\", \$2/1e6}' /proc/meminfo")"
  done
}
memoria() { for n in $NODOS; do echo -n "$n $(ssh $n "awk '/MemAvailable/{printf \"%.1f GB\", \$2/1e6}' /proc/meminfo") "; done; }
b12x_cache() {  # $1 log de arranque: lo medido al arrancar (0 = todo de la caché) y lo que salió de ella
  python3 - "$1" <<'PY'
import re,sys
L=[l for l in open(sys.argv[1], errors="replace") if l.startswith("b12x ")]
n=lambda pat,ls: max((int(m.group(1)) for l in ls for m in [re.search(pat,l)] if m), default=0)
gp=[l for l in L if "gdn_prefill" in l]
print(f"medidos máx {n(r'(\d+) measured',L)}, en caché máx {n(r'(\d+) cached',L)}, en caché con gdn_prefill {n(r'(\d+) cached',gp)}, "
      f"'tuning stopped' {sum('tuning stopped' in l for l in L)} líneas")
PY
}
mide() {  # $1 modo: tools, decode c1/c4/c8/c16 y prefill, igual que mide_todo de cadena-932.sh
  local T=$W/$1; mkdir -p $T
  python3 $D/tools-check.py $R2 $T/tools > $T/tools.log 2>&1; log "$1 tools exit=$? $(tail -1 $T/tools.log | cut -c1-100)"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $R2 --model $MODEL --concurrency 1,4,8,16 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 \
    --long-tokens 32000 --max-model-len 262144 --seed 496368 --out $T/decode.json > $T/decode.log 2>&1
  log "$1 decode $(python3 $O/resumen-decode.py $T/decode.json)"
  bash $O/prefill.sh $R2 $T/prefill > $T/prefill.log 2>&1; log "$1 prefill $(tr '\n' ' ' < $T/prefill.log)"
}
arranca() {  # $1 modo → 0 con el head de prueba sirviendo en R2
  baja_pruebas; compacta
  for d in $HEAD $WORKER; do prueba $W/$d-antes.json $1 | kubectl apply -f - >/dev/null || { log "FALLO: no pude crear la prueba $1 de $d"; return 1; }; done
  sleep 60
  for p in b2m-head b2m-worker; do
    local e; e=$(kubectl get pods -n llm -l app=$p -o jsonpath='{.items[0].spec.containers[0].env[?(@.name=="QWEN_GDN_MIX")].value} {.items[0].spec.volumes[?(@.name=="xdg")].hostPath.path}')
    [ "$e" = "$1 /srv/ursu-cache/2x-b2" ] || { log "FALLO: $p no arrancó en $1 con 2x-b2 ($e)"; return 1; }
  done
  log "$1: b2m-head y b2m-worker arrancando (autoajuste encendido, caché 2x-b2, KV $KV_PRUEBA)"
  local t0=$(date +%s) ult=0 rs
  until sano $R2; do
    rs=$(kubectl get pods -n llm -l dgx-932=medida -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' 2>/dev/null)
    [ -z "$(echo $rs | tr -d ' 0')" ] || { log "FALLO: un pod de prueba $1 se reinició ($rs)"; return 1; }
    [ $(( $(date +%s)-t0 )) -gt 3600 ] && { log "FALLO: $1 sin servir en 60 min"; return 1; }
    [ $(( $(date +%s)-ult )) -ge 300 ] && { ult=$(date +%s); log "  esperando $1 ($(( (ult-t0)/60 )) min): $(kubectl logs -n llm deploy/b2m-head --tail=50 2>/dev/null | grep -v shm_broadcast | tail -1 | cut -c1-150)"; }
    sleep 15
  done
  mkdir -p $W/$1; kubectl logs -n llm deploy/b2m-head --tail=-1 > $W/$1/arranque.log 2>&1
  log "$1 sirviendo en $(( ($(date +%s)-t0)/60 )) min; $(grep -hoE 'GPU KV cache size: [0-9,]+ tokens' $W/$1/arranque.log | tail -1)"
  log "$1 b12x al arrancar: $(b12x_cache $W/$1/arranque.log)"
}
vuelta() {
  trap - EXIT
  log "VUELTA: inicio"
  for d in b2m-head b2m-worker; do kubectl logs -n llm deploy/$d --tail=-1 > $W/$d-ultimo.log 2>&1; done
  baja_pruebas; log "VUELTA: pruebas fuera"
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

log "DGX-932 medida B2 calibrado frente a A en el TP=2 ($MODOS): espero a ArgoCD quieto, árbitro en ready y los dos servidores listos"
t0=$(date +%s)
until argo_quieto && [ "$(fase)" = ready ] && listo_head && sano $R1; do
  [ $(( $(date +%s)-t0 )) -gt 3600 ] && { log "FALLO: sin las condiciones de partida en 1 h ($(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase}|{.status.sync.status}|{.status.health.status}'), árbitro $(fase))"; exit 1; }
  sleep 30
done
[ "$(kubectl get pods -n llm -l dgx-932 --no-headers 2>/dev/null)" = "" ] || { log "FALLO: ya hay pods de prueba de DGX-932"; exit 1; }
for d in $HEAD $WORKER; do kubectl get deploy -n llm $d -o json > $W/$d-antes.json; done
for d in $VOZ; do kubectl get deploy -n llm $d -o jsonpath='{.spec.replicas}' > $W/$d.replicas; done
for d in $HEAD $WORKER; do prueba $W/$d-antes.json B2 > /dev/null || { log "FALLO: producción no está como espero (ver arriba)"; exit 1; }; done
pausa || { log "FALLO: no pude pausar el árbitro"; exit 1; }
trap vuelta EXIT
log "árbitro en sync-paused ($FIRMA)"
autosync_off; log "auto-sync de root y ai apagado"
for d in $VOZ; do kubectl scale deploy -n llm $d --replicas=0 >/dev/null; done
kubectl scale deploy -n llm $HEAD $WORKER --replicas=0 >/dev/null
until [ -z "$(kubectl get pods -n llm --no-headers 2>/dev/null | grep -E '^(omnivoice-tts-nvfp4|stt-turbo|qwen38-flash-next-head|qwen38-flash-next-worker)-[0-9a-f]+-[0-9a-z]+ ')" ]; do sleep 5; done
log "voz y TP=2 de producción a 0"
for n in $NODOS; do ssh $n "sudo -n rm -f /srv/ursu-cache/2x-b2/b12x-consultas.jsonl"; done
for m in $MODOS; do
  arranca $m || exit 1
  log "$m memoria libre antes de medir: $(memoria)"
  mide $m
  log "$m memoria libre después de medir: $(memoria)"
done
exit 0
