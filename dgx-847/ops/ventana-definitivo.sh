#!/usr/bin/env bash
# DGX-852 (Dani 13-10: «monta el sistema definitivo»; valla aprobada para ~1 h): una ventana con la configuración final a mano.
#   vista NVFP4 e768 (TP=3) · b12x en GEMM y MoE con la rueda port (1.3.0 + micro.py de ursu, QWEN38_B12X_URSU=1) · MTP con layout
#   w31 y vocabulario recortado del borrador (castellano) · SSM BF16 · launch mtp/b12x-ursu/launch-ursu-b12x.sh (Triton en /tmp).
# k=4 y luego k=2: arranque, tools + Claude Code y decode corto (c1/4/8, 60 s). Se queda arriba k=2 y ArgoCD sigue fuera hasta el
# repin por PR (la sync de ese repin reinicia con la config de git, sea k=2 o k=4). Si nada arranca: la vuelta de ventana-nvfp4.sh.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
W=$D/mtp/definitivo; mkdir -p $W
V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3; REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
DV=/opt/qwen38fn-mtp-dv/ids-es-K131072.txt.gz
export MODEL=qwen38-flash-next
RUN=$(date -u +%d%H%M)

if [ -z "${SIGUE:-}" ]; then
# ---- 0. ArgoCD fuera (el estado vivo es el de git: mtp/nvfp4/antes-* sirven para la vuelta) y pausa
for a in root ai; do
  kubectl get application -n argocd $a -o jsonpath='{.spec.syncPolicy}' | grep -q automated \
    || { log "definitivo: $a sin auto-sync al empezar, el vivo no es git; no sigo"; exit 1; }
done
for a in root ai; do kubectl patch application -n argocd $a --type json -p '[{"op":"remove","path":"/spec/syncPolicy/automated"}]' >/dev/null; done
baja llm-deploy-pause-def-$RUN || {
  for a in ai root; do kubectl patch application -n argocd $a --type merge -p "{\"spec\":{\"syncPolicy\":$(cat $D/mtp/nvfp4/$a-syncpolicy.json)}}" >/dev/null; done
  log "definitivo: la pausa falla; residente y ArgoCD sin tocar"; exit 1; }
else  # SIGUE=1: la ventana ya está montada (ArgoCD fuera, valla puesta); se rehace la config y se sigue
  if [ "$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')" = sync-paused ]; then
    kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null  # sigue en pausa (la deja un arranque fallido en pausa): los rangos a 0
    while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done; log "def: sigue en pausa, rangos a 0"
  else
    [ -n "${NOPAUSA:-}" ] || baja llm-deploy-pause-def-$RUN || { NOPAUSA=1; log "def: sin pausa; los tres rangos se rehacen a la vez"; }
  fi
fi

# ---- 1. piezas: launch, vocabulario del borrador, tabla PLE de la vista
kubectl patch cm -n llm qwen38-flash-next-vllm-launch --type merge \
  -p "$(python3 -c 'import json,sys; print(json.dumps({"data": {"launch.sh": open(sys.argv[1]).read()}}))' $D/mtp/b12x-ursu/launch-ursu-b12x.sh)" >/dev/null
kubectl get cm -n llm qwen38fn-mtp-draft-vocab >/dev/null 2>&1 || kubectl create configmap qwen38fn-mtp-draft-vocab -n llm \
  --from-file=mtp_draft_vocab.py=$D/mtp/draft-vocab/mtp_draft_vocab.py --from-file=ids-es-K131072.txt.gz=$D/mtp/draft-vocab/ids-es-K131072.txt.gz >/dev/null
for d in $DEPS; do
  kubectl patch deploy -n llm $d --type strategic -p '{"spec":{"template":{"spec":{
    "volumes":[{"name":"ple-table","hostPath":{"path":"/home/user/.cache/qwen38fn-ple-ursu","type":"DirectoryOrCreate"}},
               {"name":"mtp-draft-vocab","configMap":{"name":"qwen38fn-mtp-draft-vocab"}}],
    "containers":[{"name":"vllm","volumeMounts":[{"name":"mtp-draft-vocab","mountPath":"/opt/qwen38fn-mtp-dv","readOnly":true}]}]}}}}' >/dev/null
  kubectl set env -n llm deploy/$d -c ple-table CKPT_DIR=$V CKPT_REVISION=$REV >/dev/null
  kubectl set env -n llm deploy/$d -c vllm MODEL_PATH=$V-e768 MODEL_REVISION=$REV VLLM_REFUSAL_PROBE=0 QWEN38_B12X=1 QWEN38_B12X_MOE=1 \
    QWEN38_B12X_URSU=1 QWEN38_B12X_MTP31=1 QWEN38_B12X_MOE_WA- CUDA_LAUNCH_BLOCKING- VLLM_MTP_DRAFT_VOCAB=$DV >/dev/null
done

prueba() {  # $1 = k: arranque con resume, tools + Claude Code y decode corto
  local k=$1 T=$W/k$1; mkdir -p $T
  for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm "SPECULATIVE_CONFIG=$(spec_k $k)" >/dev/null; done
  # sin pausa: los tres rangos a 0 y a 1 a la vez (rehace en residente-lib.sh)
  local ok=0
  if [ -n "${NOPAUSA:-}" ]; then rehace && listo 5400 && ok=1
  else arriba_en_pausa 5400 && { log "def k=$k: resume del árbitro"; hook llm-deploy-resume llm-deploy-resume-def-k$k-$RUN; listo 900; } && ok=1; fi
  kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $T/arranque.log 2>&1
  [ $ok = 1 ] || { kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm --previous >> $T/arranque.log 2>&1; log "def k=$k: NO arranca"; return 1; }
  grep -E 'QWEN38_B12X|QWEN_MTP_DRAFT_VOCAB|Graph capturing finished|init engine' $T/arranque.log | sed 's/^.*\] //' | cut -c1-150
  python3 $D/tools-check.py $RES $T > $T/tools.log 2>&1; log "def k=$k: tools exit=$? $(tail -1 $T/tools.log | cut -c1-120)"
  local C; C=$(mktemp -d /tmp/dgx852-cc-XXXX); echo "hola desde DGX-852" > $C/entrada.txt
  ( cd $C && ANTHROPIC_BASE_URL=$RES ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN= CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 600 \
    claude -p --model $MODEL --allowedTools "Read Write Bash(ls:*)" --permission-mode acceptEdits --output-format json \
    "Lee entrada.txt, escribe su contenido en mayúsculas en salida.txt y luego lista el directorio con ls." ) > $T/claude-code.json 2>&1
  log "def k=$k: claude-code salida.txt=$(cat $C/salida.txt 2>/dev/null || echo NO)"; rm -rf $C
  espera_vacio
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $T/decode-corto.json > $T/decode-corto.log 2>&1
  log "def k=$k: decode $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; print(" · ".join(f"c{c} {v[\"agg_tok_s_median\"]:.0f}" for c, v in s.items()), "· acept", s["1"].get("mtp_accept_by_pos_median"))' $T/decode-corto.json 2>&1 | cut -c1-200)"
}

# ---- 2. k=4 y k=2 (se queda k=2)
prueba 4 || { log "def k=4: sigo con k=2"; NOPAUSA=1; }
[ -n "${NOPAUSA:-}" ] || baja llm-deploy-pause-def-k2-$RUN || { NOPAUSA=1; log "def: la pausa antes de k=2 falla; los tres rangos se rehacen a la vez"; }
if prueba 2; then
  log "DEFINITIVO arriba (k=2) con ArgoCD fuera y la valla puesta: falta quitar la valla, la PR a la línea del pin y el repin"
else
  log "definitivo: k=2 no arranca; vuelta a git"; DESDE=x VUELTA=1 exec bash $O/ventana-nvfp4.sh
fi
