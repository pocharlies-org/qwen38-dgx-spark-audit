#!/usr/bin/env bash
# DGX-852: arreglo del MoE b12x en TP=3 (mtp/b12x-moe-fix/LISTO.md). Espera al diagnóstico de A con CUDA_LAUNCH_BLOCKING=1,
# carga el launch.sh con el bloque QWEN38_B12X_MOE_WA y arranca A con cutover0 (la banda micro del MoE nvfp4 de b12x 1.3.0
# pasa al kernel dynamic) y el layout w31 del MoE w4a16 del MTP. Sin pausa: el árbitro queda en error tras el diagnóstico
# (Recreate rehace los tres rangos). Si arranca, la ventana sigue por DESDE: batería, B con el vocabulario y la vuelta.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
W=$D/mtp/nvfp4; V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3; RUN=$(date -u +%d%H%M)
[ -n "${SIN_DIAG:-}" ] || diag_espera a  # Dani 13-10 17:45Z: arreglar primero; el diagnóstico se corta
kubectl patch cm -n llm qwen38-flash-next-vllm-launch --type merge \
  -p "$(python3 -c 'import json,sys; print(json.dumps({"data": {"launch.sh": open(sys.argv[1]).read()}}))' $W/launch-nvfp4-wa.sh)" >/dev/null
for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm MODEL_PATH=$V-e768 QWEN38_B12X=1 QWEN38_B12X_MOE=1 QWEN38_B12X_MOE_WA=cutover0 \
  QWEN38_B12X_MTP31=1 CUDA_LAUNCH_BLOCKING- VLLM_MTP_DRAFT_VOCAB- "SPECULATIVE_CONFIG=$(spec_k 2)" >/dev/null; done
log "a arreglada: env puesta a mano (e768 + MoE b12x con cutover0 + MTP w31)"; sleep 60
DESDE="$V-e768 1 1 a-arreglada" exec bash $O/ventana-nvfp4.sh
