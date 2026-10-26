#!/usr/bin/env bash
# DGX-852 (Dani, 13-10 18:40Z): la calidad de A pasa detrás de B. Cuando la ventana en curso acaba la caché a 64K de A
# arreglada, se corta (antes o justo al empezar la calidad) y sigue la ventana nueva por B con SOLO_B=1; la calidad de A va al final.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
P=$1  # PID exacto de la ventana en curso
until [ "$(grep -c 'A caché 64K: {"paso": "3' $O/ventana-nvfp4.log)" -ge 2 ]; do sleep 5; done
pkill -P $P; kill $P
sleep 2; for q in $(ps -eo pid=,args= | awk '/launch-benchy-quality.py|run_battery.py/ && !/awk/{print $1}'); do kill $q; done
log "calidad de A detrás de B (Dani): ventana cortada tras la caché de A arreglada; sigue por B"
DESDE="/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3-e768 1 1 a-arreglada" SOLO_B=1 exec bash $O/ventana-nvfp4.sh
