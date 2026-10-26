#!/usr/bin/env bash
# DGX-852: la 1x de E2 falló en el primer intento (VLLM_GDN_COMPACT_RECORDS=1 exige el GDN b12x) y prueba-e2.sh la dio por
# perdida. Con la llave a 0: espera a que acabe prueba-e2.sh (que mide la 2x sola), luego la 1x sola y los dos a la vez.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
T=$D/mtp/definitivo/e2; R2=http://spark2:8000; R1=http://spark3:8000; export MODEL=qwen3.8-flash-next
while pgrep -f 'bash prueba-e2.sh' >/dev/null; do sleep 20; done
t0=$(date +%s)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R1/health)" = 200 ]; do
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "e2: TP=1 sin health en 60 min"; exit 1; }; sleep 15
done
kubectl logs -n llm deploy/ursu-1x > $T/tp1-arranque.log 2>&1
log "e2: tp1 arriba; $(grep -hE 'attention block size|Model loading took|Available KV cache memory|GPU KV cache size|draft vocab|DRAFT_VOCAB' $T/tp1-arranque.log | sed 's/^.*\] //' | cut -c1-100 | sort -u | paste -sd' ')"
mide() {
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $1 --model $MODEL --concurrency $3 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $2.json > $2.log 2>&1
}
mkdir -p $T/tp1; python3 $D/tools-check.py $R1 $T/tp1 > $T/tp1-tools.log 2>&1; log "e2: TP=1 tools exit=$? $(tail -1 $T/tp1-tools.log | cut -c1-100)"
mide $R1 $T/tp1-decode 1,4,8; log "e2: TP=1 decode $(python3 $O/resumen-decode.py $T/tp1-decode.json)"
[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R2/health)" = 200 ] || { log "e2: la 2x no está arriba; sin prueba conjunta"; exit 1; }
for c in 4 8; do
  mide $R2 $T/junto-c$c-tp2 $c & p=$!; mide $R1 $T/junto-c$c-tp1 $c; wait $p
  log "e2: juntos c$c+c$c: TP=2 $(python3 $O/resumen-decode.py $T/junto-c$c-tp2.json | cut -d' ' -f1-2) · TP=1 $(python3 $O/resumen-decode.py $T/junto-c$c-tp1.json | cut -d' ' -f1-2)"
done
log "e2: hecho (2x y 1x arriba con E2; residente a 0, árbitro en pausa)"
