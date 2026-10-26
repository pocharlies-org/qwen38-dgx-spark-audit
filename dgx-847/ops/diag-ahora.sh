#!/usr/bin/env bash
# DGX-852 (Dani, 13-10 ~16:40Z: «quiero arreglar el problema», no hace falta pasar todas las pruebas): el diagnóstico de A
# se adelanta. Cuando A1 termine decode, benchy y la caché a 64K, se corta la ventana (la calidad y B quedan para la
# configuración arreglada), se arranca A con CUDA_LAUNCH_BLOCKING=1 hasta que falle, y se vuelve a A1 para que el residente
# sirva mientras se prepara el arreglo. ArgoCD sigue fuera y la valla puesta.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
W=$D/mtp/nvfp4; V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3; RUN=$(date -u +%d%H%M)
until grep -q 'A caché 64K: {"paso": "3' $O/ventana-nvfp4.log; do sleep 10; done
P=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="ventana-nvfp4.sh"{print $1}')
[ -n "$P" ] && { pkill -P $P; kill $P; }
sleep 2; for q in $(ps -eo pid=,args= | awk '/launch-benchy-quality.py|run_battery.py/ && !/awk/{print $1}'); do kill $q; done
log "ventana cortada tras la caché de A1: la calidad y B quedan para la configuración arreglada"
diag a $V-e768 1 1
for d in $DEPS; do kubectl set env -n llm deploy/$d -c vllm MODEL_PATH=$V QWEN38_B12X=1 QWEN38_B12X_MOE- CUDA_LAUNCH_BLOCKING- >/dev/null; done
log "a1: de vuelta a mano tras el diagnóstico"; sleep 60
listo 5400 && log "diag: fin (A1 sirviendo; ArgoCD fuera y valla puesta)" || log "diag: fin, pero A1 no vuelve"
