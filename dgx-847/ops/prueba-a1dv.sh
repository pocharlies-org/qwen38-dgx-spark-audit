#!/usr/bin/env bash
# DGX-852 (14-10 ~23:45Z): el MoE b12x en TP=3 con el micro de ursu arranca pero es más lento que A1 (c1 38 frente a 55). Candidato
# definitivo: A1 (vista 640 con EP, b12x solo en los GEMM, SSM BF16) + vocabulario recortado del borrador (castellano), k=2. Con el árbitro
# en pausa (sin su plazo de 30 min ni el resume de 15): rangos a 0, env, los tres a 1 a la vez, herramientas, Claude Code y decode corto.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3; DV=/opt/qwen38fn-mtp-dv/ids-es-K131072.txt.gz; K=${K:-2}; T=$D/mtp/definitivo/a1dv-k$K; mkdir -p $T
export MODEL=qwen38-flash-next
kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm MODEL_PATH=$V QWEN38_B12X=1 QWEN38_B12X_MOE- QWEN38_B12X_URSU- QWEN38_B12X_MTP31- \
  QWEN38_B12X_MOE_WA- VLLM_MTP_DRAFT_VOCAB=$DV "SPECULATIVE_CONFIG=$(spec_k $K)" >/dev/null; done
log "a1dv k=$K: A1 + vocabulario, k=$K (árbitro en pausa)"
arriba_en_pausa 5400 || { kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm --previous > $T/arranque.log 2>&1; log "a1dv k=$K: NO arranca"; exit 1; }
kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $T/arranque.log 2>&1
python3 $D/tools-check.py $RES $T > $T/tools.log 2>&1; log "a1dv k=$K: tools exit=$? $(tail -1 $T/tools.log | cut -c1-120)"
C=$(mktemp -d /tmp/dgx852-cc-XXXX); echo "hola desde DGX-852" > $C/entrada.txt
( cd $C && ANTHROPIC_BASE_URL=$RES ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN= CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 600 \
  claude -p --model $MODEL --allowedTools "Read Write Bash(ls:*)" --permission-mode acceptEdits --output-format json \
  "Lee entrada.txt, escribe su contenido en mayúsculas en salida.txt y luego lista el directorio con ls." ) > $T/claude-code.json 2>&1
log "a1dv k=$K: claude-code salida.txt=$(cat $C/salida.txt 2>/dev/null || echo NO)"; rm -rf $C
python3 $TOOLS/qwen38_decode_sweep.py --base-url $RES --model qwen38-flash-next --concurrency 1,4,8 --warmup 20 --duration 60 --reps 1 \
  --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
  --max-model-len 262144 --seed 496368 --out $T/decode-corto.json > $T/decode-corto.log 2>&1
log "a1dv k=$K: decode $(python3 $O/resumen-decode.py $T/decode-corto.json)"
