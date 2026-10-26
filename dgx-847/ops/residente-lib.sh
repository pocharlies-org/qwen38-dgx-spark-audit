# Funciones de ventana del residente qwen38 TP=3 (se cargan con `. residente-lib.sh`); las mismas de mtp-cadena.sh.
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; O=$D/ops; RES=http://tailnet-node:8000
DEPS="qwen38-flash-next-head qwen38-flash-next-worker qwen38-flash-next-worker-dgx3"
BENCHY=/home/user/bin/benchy-venv/bin/llama-benchy; TOOLS=/home/user/bench-results/dgx-768/tools
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
hook() {  # $1 = job de ArgoCD a clonar (llm-deploy-pause|llm-deploy-resume), $2 = nombre nuevo; espera a que acabe
  local src=$1  # si ArgoCD ya borró la plantilla (pasó el 13-10 a las 17:22Z), la copia más reciente: mismo spec
  kubectl get job -n llm $src >/dev/null 2>&1 || src=$(kubectl get job -n llm --no-headers -o custom-columns=N:.metadata.name,C:.metadata.creationTimestamp \
    | awk -v p="$1-" 'index($1, p) == 1' | sort -k2 | tail -1 | cut -d' ' -f1)
  kubectl get job -n llm $src -o json | python3 -c '
import json,sys; j=json.load(sys.stdin); t=j["spec"]["template"]; l=t["metadata"].get("labels",{})
for k in ("controller-uid","batch.kubernetes.io/controller-uid","job-name","batch.kubernetes.io/job-name"): l.pop(k,None)
print(json.dumps({"apiVersion":"batch/v1","kind":"Job","metadata":{"name":sys.argv[1],"namespace":"llm","labels":{"app":"llm-deploy-hooks"}},
 "spec":{"backoffLimit":j["spec"].get("backoffLimit",2),"ttlSecondsAfterFinished":86400,"template":t}}))' $2 | kubectl create -f - >/dev/null
  local t0=$(date +%s) st  # kubectl wait --for=Complete no vuelve si el Job falla: se mira succeeded/failed
  while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do
    st=$(kubectl get job -n llm $2 -o jsonpath='{.status.succeeded}/{.status.failed}')
    case $st in 1/*) log "job $2 completo"; return 0 ;; */[1-9]*) break ;; esac
    sleep 10
  done
  log "job $2 NO completo: $(kubectl logs -n llm job/$2 --tail=2 2>&1 | tr '\n' ' ' | cut -c1-200)"; return 1
}
arbitro_ready() {  # la pausa se niega si el árbitro no está en ready (13-10 15:32Z: phase=recovering 2 min); espera hasta 15 min
  local t0=$(date +%s) p
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    p=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
    [ "$p" = ready ] && return 0; sleep 15
  done; log "árbitro sin ready en 15 min (phase=$p)"; return 1
}
listo() {  # residente 1/1 ×3, árbitro ready, /health 200 (tope $1 s, 60 min por defecto); un reinicio de contenedor = fallo
  local t0=$(date +%s) cap=${1:-3600}
  while [ $(( $(date +%s) - t0 )) -lt $cap ]; do
    p=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
    d=$(kubectl get deploy -n llm $DEPS -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
    h=$(curl -s -o /dev/null -w '%{http_code}' -m 5 $RES/health || true)
    [ "$h" = 200 ] && [ "$p" = ready ] && [ "$d" = "1/1 1/1 1/1 " ] && { log "residente listo en $(( ($(date +%s) - t0) / 60 )) min"; return 0; }
    # el pod viejo que sale (Terminating) no cuenta; un worker que arranca antes que el head nuevo reinicia una vez (gloo
    # «connection refused», 13-10 17:44Z): fallo = el head reinicia o un worker reinicia dos veces
    kubectl get pods -n llm --no-headers | awk '/^qwen38-flash-next-/ && $3 != "Terminating" && ($4 >= 2 || ($1 ~ /-head-/ && $4 >= 1)) {f=1} END{exit !f}' \
      && { log "residente: un rango se ha reiniciado (arb=$p deploys=[$d])"; return 1; }
    sleep 30
  done; log "residente sin listo en $(( cap / 60 )) min (arb=$p deploys=[$d] health=$h)"; return 1
}
ajenas() { curl -s -m 5 $RES/metrics | awk '/^vllm:num_requests_running/{s+=$2} END{print int(s)}'; }
espera_vacio() {  # con la valla, las sesiones ya ligadas a local acaban solas: espera a 0 peticiones ajenas (tope 20 min)
  local t0=$(date +%s) n
  while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do n=$(ajenas); [ "$n" = 0 ] && { sleep 20; [ "$(ajenas)" = 0 ] && return 0; }; sleep 20; done
  log "sigue habiendo tráfico ajeno ($(ajenas) en curso) tras 20 min: mido igual; la sonda lo marca"
}
arriba_en_pausa() {  # con el árbitro en pausa: los tres rangos a 1 a la vez y espera health 200 + 1/1 ×3 (tope $1 s). En pausa el árbitro
  # no repara: una primera compilación de más de 30 min no la corta su READY_TIMEOUT (13-10 21:55Z). El resume va después.
  local t0=$(date +%s) cap=${1:-5400} d h
  kubectl scale deploy -n llm $DEPS --replicas=1 >/dev/null; log "arriba en pausa: los tres rangos a 1 a la vez"
  while [ $(( $(date +%s) - t0 )) -lt $cap ]; do
    d=$(kubectl get deploy -n llm $DEPS -o jsonpath='{range .items[*]}{.status.readyReplicas}/{.spec.replicas} {end}')
    h=$(curl -s -o /dev/null -w '%{http_code}' -m 5 $RES/health || true)
    [ "$h" = 200 ] && [ "$d" = "1/1 1/1 1/1 " ] && { log "arriba en pausa en $(( ($(date +%s) - t0) / 60 )) min"; return 0; }
    kubectl get pods -n llm --no-headers | awk '/^qwen38-flash-next-/ && $3 != "Terminating" && ($4 >= 2 || ($1 ~ /-head-/ && $4 >= 1)) {f=1} END{exit !f}' \
      && { log "arriba en pausa: un rango se ha reiniciado (deploys=[$d])"; return 1; }
    sleep 30
  done; log "arriba en pausa: sin listo en $(( cap / 60 )) min"; return 1
}
rehace() {  # sin pausa del árbitro: los tres rangos a 0 hasta que no quede ningún pod (su vigilante puede subir alguno entre medias) y
  # luego los tres a 1 a la vez; borrar pods o el Recreate de un set env los desparejan (gotcha_tp3_rango_nuevo_conecta_al_tcpstore…)
  local t0=$(date +%s)
  while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do
    kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null; sleep 5
    [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "rehace: siguen pods tras 10 min"; return 1; }
  done
  kubectl scale deploy -n llm $DEPS --replicas=1 >/dev/null; log "rehace: los tres rangos a 1 a la vez"
}
baja() {  # pausa del árbitro (Job clonado $1) y los tres rangos a 0 (TP=3 no se reinicia por partes); sin pausa no se baja nada
  arbitro_ready || return 1
  hook llm-deploy-pause $1 || return 1
  kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null
  while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
  for h in dgx dgx2 dgx3; do ssh $h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
}
spec_k() { echo "{\"method\":\"mtp\",\"num_speculative_tokens\":$1,\"disable_eagle_block_drop\":true,\"num_speculative_tokens_per_batch_size\":[[1,16,$1]],\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"probabilistic\"}"; }
mide() {  # $1 = directorio, $2 = etiqueta: decode v4 (c1/4/8/10 ×3) y benchy medium, como en mtp-cadena.sh
  local M=$1 t=$2; mkdir -p $M
  espera_vacio; log "$t: mido (ajenas en curso: $(ajenas))"
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8,10 --warmup 30 --duration 120 --reps 3 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $M/decode-agg.json > $M/decode-agg.log 2>&1
  log "$t: decode exit=$?"; tr '\r' '\n' < $M/decode-agg.log | grep -E '^C= ' | cut -c1-120
  $BENCHY --base-url $RES/v1 --model qwen38-flash-next --served-model-name qwen38-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid \
    --pp 2048 --tg 128 --depth 0 65535 --concurrency 1 4 8 --runs 3 --enable-prefix-caching --format csv --save-result $M/medium.csv > $M/benchy.log 2>&1
  log "$t: benchy exit=$?"
}
diag() {  # (usa $W y $RUN del guion) $1 etiqueta, $2 vista, $3 b12x, $4 moe: arranque con CUDA_LAUNCH_BLOCKING=1 para que el acceso ilegal salte en el
  # kernel que lo provoca y no en el allreduce siguiente (Dani, 13-10). Con el árbitro en error no hay pausa: Recreate rehace los rangos.
  local T=$W/diag-$1 t0 r="" p; mkdir -p $T
  local e=(MODEL_PATH=$2 "SPECULATIVE_CONFIG=$(spec_k 2)" VLLM_MTP_DRAFT_VOCAB- CUDA_LAUNCH_BLOCKING=1)
  [ $3 = 1 ] && e+=(QWEN38_B12X=1) || e+=(QWEN38_B12X-)
  [ $4 = 1 ] && e+=(QWEN38_B12X_MOE=1) || e+=(QWEN38_B12X_MOE-)
  p=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
  if [ "$p" = ready ] && baja llm-deploy-pause-nvfp4-diag-$1-$RUN; then
    for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm "${e[@]}" >/dev/null; done
    hook llm-deploy-resume llm-deploy-resume-nvfp4-diag-$1-$RUN
  else
    for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm "${e[@]}" >/dev/null; done
  fi
  log "diag $1: arranque con CUDA_LAUNCH_BLOCKING=1 (árbitro en $p)"; sleep 90
  diag_espera $1
}
diag_espera() {  # $1 etiqueta: espera al fallo (o al /health) del arranque de diagnóstico y resume la traza
  local T=$W/diag-$1 t0=$(date +%s) r=""
  while [ $(( $(date +%s) - t0 )) -lt 7200 ]; do
    for d in $DEPS; do { kubectl logs -n llm deploy/$d -c vllm --previous 2>/dev/null; kubectl logs -n llm deploy/$d -c vllm 2>&1; } > $T/$d.log; done
    grep -qlE 'illegal memory access|CUDA error' $T/*.log && { r="falla"; break; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $RES/health)" = 200 ] && { r="arranca sin fallo"; break; }
    sleep 60
  done
  sleep 30; for d in $DEPS; do { kubectl logs -n llm deploy/$d -c vllm --previous 2>/dev/null; kubectl logs -n llm deploy/$d -c vllm 2>&1; } > $T/$d.log; done
  for f in $T/*.log; do grep -vE 'GET /metrics|shm_broadcast|import_utils' $f | grep -m1 -B60 -E 'illegal memory access|CUDA error' \
    | grep -E 'File "|Error|error' | grep -vE 'torch/nn/modules/module.py|_contextlib.py' | sed "s|^|$(basename $f .log): |" | cut -c1-260; done > $T/resumen.txt
  log "diag $1: ${r:-sin resultado en 2 h} · $T/resumen.txt"; grep -E 'b12x|fused_moe|File' $T/resumen.txt | tail -8
}
