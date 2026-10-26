#!/usr/bin/env bash
# DGX-852, ventana en el residente con la VALLA puesta a mano antes (panel model-routing → alibaba), tras nvfp4-previo.sh:
#  A  el checkpoint NVFP4 de ursu en vista TP=3 con los expertos rellenos a 768 + b12x en el MoE (sin EP) y en los GEMM densos;
#     SSM en bf16 y GDN como hoy; MTP k=2. Sonda de rechazo apagada: su calibración es de los pesos FP8.
#     Batería: tools (+ Claude Code si hay /v1/messages), decode v4, benchy medium, caché 64K (prueba A); calidad 7 suites tras B.
#     Si A no arranca: A1 = vista 640 con EP y b12x solo en los GEMM; si tampoco: A2 = vista 640 sin b12x.
#  B  lo que haya arrancado en A + vocabulario recortado del borrador (lista con castellano), k=2, 3 y 4: decode v4 y benchy.
#     La calidad de A va detrás de B, con A otra vez arriba.
#  vuelta: Deployments y launch.sh como estaban al empezar (copia en mtp/nvfp4/antes-*), ConfigMap del vocabulario fuera,
#     auto-sync de root y ai, resume. La VALLA se quita a mano al acabar.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
W=$D/mtp/nvfp4; mkdir -p $W
V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3; REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
DV=/opt/qwen38fn-mtp-dv/ids-es-K131072.txt.gz
export MODEL=qwen38-flash-next
RUN=$(date -u +%d%H%M)  # sufijo de los Jobs de pausa/resume: una segunda corrida no reutiliza un Job ya completo

# DESDE="<vista> <b12x> <moe> <etiqueta>": la ventana ya está montada (pasos 0 y 1 hechos, copia antes-* del inicio) y esa
# configuración se puso a mano porque el árbitro quedó en error (13-10: A rompe al capturar los grafos y la pausa lo rechaza).
if [ -z "${DESDE:-}" ]; then
# ---- 0. copia del estado vivo (= git) y ArgoCD fuera
for a in root ai; do
  kubectl get application -n argocd $a -o jsonpath='{.spec.syncPolicy}' > $W/$a-syncpolicy.json
  grep -q automated $W/$a-syncpolicy.json || { log "$a sin auto-sync al empezar: el estado vivo no es el de git, no sigo"; exit 1; }
done
log "inicio: ai $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status}/{.status.health.status}')"
kubectl get deploy -n llm $DEPS -o json > $W/antes-deploys.json
kubectl get cm -n llm qwen38-flash-next-vllm-launch -o json > $W/antes-launch-cm.json
for a in root ai; do kubectl patch application -n argocd $a --type json -p '[{"op":"remove","path":"/spec/syncPolicy/automated"}]' >/dev/null; done
baja llm-deploy-pause-nvfp4-$RUN || {  # sin pausa no se toca nada: ArgoCD como estaba y fuera
  for a in ai root; do kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $W/$a-syncpolicy.json)}}" >/dev/null; done
  log "NVFP4: no empieza (la pausa del árbitro falla); residente y ArgoCD sin tocar"; exit 1; }

# ---- 1. piezas comunes: launch.sh con los bloques b12x y vocabulario (los dos inertes sin su env), la vista y su tabla PLE
kubectl patch cm -n llm qwen38-flash-next-vllm-launch --type merge \
  -p "$(python3 -c 'import json,sys; print(json.dumps({"data": {"launch.sh": open(sys.argv[1]).read()}}))' $W/launch-nvfp4.sh)" >/dev/null
kubectl create configmap qwen38fn-mtp-draft-vocab -n llm --from-file=mtp_draft_vocab.py=$D/mtp/draft-vocab/mtp_draft_vocab.py \
  --from-file=ids-es-K131072.txt.gz=$D/mtp/draft-vocab/ids-es-K131072.txt.gz >/dev/null
kubectl label cm -n llm qwen38fn-mtp-draft-vocab dgx-852=ventana >/dev/null
for d in $DEPS; do
  kubectl patch deploy -n llm $d --type strategic -p '{"spec":{"template":{"spec":{
    "volumes":[{"name":"ple-table","hostPath":{"path":"/home/user/.cache/qwen38fn-ple-ursu","type":"DirectoryOrCreate"}},
               {"name":"mtp-draft-vocab","configMap":{"name":"qwen38fn-mtp-draft-vocab"}}],
    "containers":[{"name":"vllm","volumeMounts":[{"name":"mtp-draft-vocab","mountPath":"/opt/qwen38fn-mtp-dv","readOnly":true}]}]}}}}' >/dev/null
  kubectl set env -n llm deploy/$d -c ple-table CKPT_DIR=$V CKPT_REVISION=$REV >/dev/null
  kubectl set env -n llm deploy/$d -c vllm MODEL_REVISION=$REV VLLM_REFUSAL_PROBE=0 >/dev/null
done

fi  # DESDE

arranca() {  # $1 etiqueta, $2 vista, $3 QWEN38_B12X, $4 QWEN38_B12X_MOE, $5 k, $6 lista del borrador (vacía = cabeza entera)
  local e=(MODEL_PATH=$2 "SPECULATIVE_CONFIG=$(spec_k $5)" CUDA_LAUNCH_BLOCKING-) ok=0
  [ $3 = 1 ] && e+=(QWEN38_B12X=1) || e+=(QWEN38_B12X-)
  [ $4 = 1 ] && e+=(QWEN38_B12X_MOE=1) || e+=(QWEN38_B12X_MOE-)
  [ -n "${6:-}" ] && e+=(VLLM_MTP_DRAFT_VOCAB=$6) || e+=(VLLM_MTP_DRAFT_VOCAB-)
  for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm "${e[@]}" >/dev/null; done
  log "$1: resume del árbitro"
  hook llm-deploy-resume llm-deploy-resume-nvfp4-$1-$RUN
  listo 5400 && ok=1  # el primer arranque con b12x compila sus kernels: 90 min de tope
  kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $W/$1-arranque.log 2>&1
  [ $ok = 1 ] || kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm --previous >> $W/$1-arranque.log 2>&1
  grep -E 'QWEN38_B12X|QWEN_MTP_DRAFT_VOCAB|MoE backend|Selected .*Kernel|attention block size|GPU KV cache size|Error|ABORTA' $W/$1-arranque.log \
    | sed 's/^.*\] //' | sort | uniq -c | sort -rn | head -14 | cut -c1-200
  return $(( 1 - ok ))
}

if [ -z "${VUELTA:-}" ]; then  # VUELTA=1 (con DESDE): solo la vuelta (Dani 13-10 19:10Z: basta de pruebas)
# ---- 2. A (con sus dos vueltas atrás)
A=""
if [ -n "${DESDE:-}" ]; then  # sin pausa: el árbitro en error la rechaza; Recreate rehace los tres rangos al cambiar el env
  set -- $DESDE
  if listo 5400; then A="$1 $2 $3"; kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $W/$4-arranque.log 2>&1
    [ "$4" = a1 ] && DIAG="a $V-e768 1 1"  # A1 arranca: el culpable de A es el MoE b12x (o el relleno a 768); se diagnostica al final
  else
    [ "$4" = a1 ] && diag a1 $V 1 0  # A1 también cae: el culpable está en los GEMM b12x
    kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm --previous > $W/$4-arranque.log 2>&1
    if [ "$4" = a1 ]; then
      for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm MODEL_PATH=$V QWEN38_B12X- QWEN38_B12X_MOE- CUDA_LAUNCH_BLOCKING- >/dev/null; done
      log "a2: env puesta a mano (vista 640 sin b12x)"; sleep 60
      if listo 5400; then A="$V 0 0"; fi
      kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $W/a2-arranque.log 2>&1
    fi
  fi
elif arranca a $V-e768 1 1 2; then A="$V-e768 1 1"
elif baja llm-deploy-pause-nvfp4-a1-$RUN; then
  if arranca a1 $V 1 0 2; then A="$V 1 0"
  elif baja llm-deploy-pause-nvfp4-a2-$RUN; then arranca a2 $V 0 0 2 && A="$V 0 0"; fi
fi
if [ -n "$A" ]; then
  T=$W/a; mkdir -p $T; log "A: arriba con [$A]"
  if [ -z "${SOLO_B:-}" ]; then  # SOLO_B=1: tools, Claude Code, velocidad y caché de A ya medidos; sigue por B
  python3 $D/tools-check.py $RES $T > $T/tools.log 2>&1; log "A: tools exit=$? $(tail -1 $T/tools.log | cut -c1-150)"
  if grep -q '"anthropic_messages": 200' $T/tools-check.json 2>/dev/null; then
    C=$(mktemp -d /tmp/dgx852-cc-XXXX); echo "hola desde DGX-852" > $C/entrada.txt
    ( cd $C && ANTHROPIC_BASE_URL=$RES ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN= CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 600 \
      claude -p --model $MODEL --allowedTools "Read Write Bash(ls:*)" --permission-mode acceptEdits --output-format json \
      "Lee entrada.txt, escribe su contenido en mayúsculas en salida.txt y luego lista el directorio con ls." ) > $T/claude-code.json 2>&1
    log "A: claude-code exit=$? salida.txt=$(cat $C/salida.txt 2>/dev/null || echo NO)"
  fi
  mide $T/k2 "A k=2"
  python3 -I $D/prueba-a-cache.py $RES $T/a-cache-ctx65535.json 65535 | sed -u 's/^/A caché 64K: /'
  fi  # SOLO_B
  # ---- 3. B: lo mismo con el vocabulario recortado
  for k in 2 3 4; do
    baja llm-deploy-pause-nvfp4-b$k-$RUN || { log "B k=$k: la pausa falla, paso a la vuelta"; break; }
    arranca b$k $A $k $DV || { log "B k=$k: no arranca, sigo"; continue; }
    mide $W/b-dv-k$k "B vocabulario k=$k"
  done
  # ---- 3b. calidad de A, detrás de B (Dani 13-10 18:40Z): A otra vez arriba, sin vocabulario y k=2
  if baja llm-deploy-pause-nvfp4-ac-$RUN && arranca a-calidad $A 2; then
    RES=$RES DEPLOY=qwen38-flash-next-head ARM=nvfp4 BASE=$T DEADLINE=$(( $(date +%s) + 6*3600 )) SKIP_BENCHY=1 TICKET=DGX-852 ROL=cto \
      python3 $D/launch-benchy-quality.py > $T/run-quality.log 2>&1
    log "A: calidad exit=$? ($(head -2 $T/DONE-benchy-quality | tr '\n' ' '))"
  else log "A: calidad sin hacer (A no vuelve a arrancar tras B)"; fi
else log "A: no arranca en ninguna de las tres formas; vuelvo"; fi

[ -n "${DIAG:-}" ] && diag $DIAG

fi  # VUELTA

# ---- 4. vuelta: estado de git (copia del inicio), ArgoCD y resume
log "vuelta: residente como estaba"
REP=0; SINPAUSA=  # sin pausa (árbitro fuera de ready): replace a 0 y rehace() sube los tres rangos a la vez con la config de git
[ "$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')" = ready ] \
  && baja llm-deploy-pause-nvfp4-fin-$RUN || { SINPAUSA=1; log "vuelta: sin pausa; los tres rangos se rehacen a la vez con la config de git"; }
REP=$REP python3 - $W/antes-deploys.json <<'PY' | kubectl replace -f - >/dev/null
import json, os, sys
l = json.load(open(sys.argv[1]))
for d in l["items"]:
    for k in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        d["metadata"].pop(k, None)
    d.pop("status", None); d["spec"]["replicas"] = int(os.environ["REP"])  # con pausa, las sube el árbitro con el resume
print(json.dumps(l))
PY
kubectl patch cm -n llm qwen38-flash-next-vllm-launch --type merge \
  -p "$(python3 -c 'import json,sys; print(json.dumps({"data": json.load(open(sys.argv[1]))["data"]}))' $W/antes-launch-cm.json)" >/dev/null
kubectl delete cm -n llm qwen38fn-mtp-draft-vocab >/dev/null
argocd() { for a in ai root; do kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $W/$a-syncpolicy.json)}}" >/dev/null; done; }
if [ -n "$SINPAUSA" ]; then  # ArgoCD vuelve con el residente listo: su PreSync de pausa no pasa con el árbitro reparando
  rehace; listo 5400; argocd
else
  argocd
  sleep 120  # si ArgoCD ve diferencias sincroniza con su PreSync de pausa; el resume va después, a mano (el PostSync no siempre corre)
  hook llm-deploy-resume llm-deploy-resume-nvfp4-fin-$RUN
  listo
fi
log "ai: $(kubectl get application -n argocd ai -o jsonpath='{.status.sync.status}/{.status.health.status}') · launch.sh $(kubectl get cm -n llm qwen38-flash-next-vllm-launch -o jsonpath='{.data.launch\.sh}' | grep -c 'QWEN38_B12X') líneas b12x"
log "NVFP4: fin (falta quitar la VALLA)"
