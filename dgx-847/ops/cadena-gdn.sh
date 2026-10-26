#!/usr/bin/env bash
# DGX-852 (Dani 14-10): los arreglos del GDN sobre E2 en la 2x. Base greedy de E2 (la 2x que está arriba, dos capturas = suelo
# de ruido), luego A0 (decode CUDA fusionado + prefill triton, sin parche) y A (decode b12x BF16 + prefill triton, parche
# gdn-mix), cada uno con su comparación greedy contra E2 y el decode c1/4/8/10. B no es servible como parche (gdn-mix/INFORME.md §4).
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
G=$D/mtp/definitivo/gdn-mix; R2=http://spark2:8000; mkdir -p $G/capturas
[ -s $G/capturas/e2-2.json ] || for n in 1 2; do
  python3 -I $G/greedy_compare.py --url $R2 --model qwen3.8-flash-next --long-tokens 16000 --conc 8 --capture $G/capturas/e2-$n.json \
    > $G/capturas/e2-$n.log 2>&1 || { log "gdn: captura E2 $n falla: $(tail -1 $G/capturas/e2-$n.log)"; exit 1; }
done; log "gdn: base greedy de E2 capturada (2)"
for m in A0 A; do
  bash $G/prueba-gdn-mix.sh $m 2 $G/capturas/e2-1.json || { log "gdn: $m no sale (ver arriba)"; continue; }
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $R2 --model qwen3.8-flash-next --concurrency 1,4,8,10 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $G/$m-decode.json > $G/$m-decode.log 2>&1
  log "gdn: $m decode $(python3 $O/resumen-decode.py $G/$m-decode.json)"
done
log "gdn: hecho"
