#!/usr/bin/env bash
# DGX-852 (14-10): la configuración de producción de k8s-ai#177 en los Deployments de prueba, antes de fusionar:
# imágenes nuevas de ursu (2x v2.1.0 k94 con el borrador D3, 1x v2.3.0 k95), receta con los ajustes de producción
# (served-model-name, priority, prompt-tokens-details, keep-alive) y el mod b12x de refusal.
# Arranque + greedy, decode (cada servidor a su vez, como cadena-tp21-A.sh), prefill en frío (los dos a la vez, como
# prefill-tp21.sh) y la batería de negativas.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
G=$D/mtp/definitivo/gdn-mix; P=$D/mtp/definitivo/prod-2mas1; RF=$D/mtp/definitivo/refusal; mkdir -p $P
R2=http://spark2:8000; R1=http://spark3:8000
( REFUSAL=1 N=P-tp2 bash $G/prueba-gdn-mix.sh A 2 $G/capturas/A-ref2-1.json > $P/arranque-tp2.log 2>&1 ) & a=$!
( REFUSAL=1 N=P-tp1 bash $G/prueba-gdn-mix.sh A 1 $G/capturas/A-ref1-1.json > $P/arranque-tp1.log 2>&1 ) & b=$!
wait $a; ea=$?; wait $b; eb=$?
log "prod-2mas1: arranque TP=2 exit=$ea · TP=1 exit=$eb"
[ $ea = 0 ] && [ $eb = 0 ] || exit 1
for s in "2 $R2 1,4,8,16" "1 $R1 1,4,8"; do set -- $s
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $2 --model qwen3.8-flash-next --concurrency $3 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 \
    --long-tokens 32000 --max-model-len 262144 --seed 496368 --out $P/decode-tp$1.json > $P/decode-tp$1.log 2>&1
  log "prod-2mas1: decode TP=$1 $(python3 $O/resumen-decode.py $P/decode-tp$1.json)"
done
bash $O/prefill.sh $R2 $P/prefill-tp2 > $P/prefill-tp2.log 2>&1 & a=$!
bash $O/prefill.sh $R1 $P/prefill-tp1 > $P/prefill-tp1.log 2>&1 & b=$!
wait $a $b; log "prod-2mas1: prefill hecho"
( cd /tmp && python3 $RF/valida-refusal.py $R2 $P/refusal-tp2 > $P/refusal-tp2.log 2>&1 ) & a=$!
( cd /tmp && python3 $RF/valida-refusal.py $R1 $P/refusal-tp1 > $P/refusal-tp1.log 2>&1 ) & b=$!
wait $a $b; log "prod-2mas1: refusal TP=2 $(tail -c 400 $P/refusal-tp2.log) · TP=1 $(tail -c 400 $P/refusal-tp1.log)"
