#!/usr/bin/env bash
# DGX-852 (Dani 14-10): su fork + estado SSM BF16 + arreglo M (A: decode b12x BF16 + prefill triton; B2: GDN b12x entero en BF16
# con registros compactos y checkpoints diferidos) + su all-reduce RoCE +
# vocabulario castellano, en TP=2 (dgx2+dgx1) y TP=1 (dgx3). Baja el TP=3. La 2x con A: greedy contra E2 TP=2, tools y decode
# hasta c16. La 1x: primero E2 (la base greedy de TP=1, no la hay) y luego A contra esa base, tools y decode. Luego los dos a la vez.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
M=${M:-A}; G=$D/mtp/definitivo/gdn-mix; T=$D/mtp/definitivo/tp21-$M; mkdir -p $T
R2=http://spark2:8000; R1=http://spark3:8000; export MODEL=qwen3.8-flash-next
F="deploy/ursu-fork-3x-head deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3"
kubectl scale -n llm $F --replicas=0 >/dev/null; while kubectl get pods -n llm -l dgx-852=fork-tp3 --no-headers | grep -q .; do sleep 5; done
log "tp21-$M: TP=3 abajo; arrancan TP=2 y TP=1 con $M"

mide() {  # $1 URL, $2 fichero sin extensión, $3 concurrencias
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $1 --model $MODEL --concurrency $3 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $2.json > $2.log 2>&1
}
dos() {
  # 14-10 11:15Z: stt-turbo (residente de llm-tp en dgx2) despertó durante el arranque y con 0.8 el head se quedó sin memoria
  N=$M-r2 EXTRA="--gpu-memory-utilization ${UTIL2:-0.75}" bash $G/prueba-gdn-mix.sh $M 2 $G/capturas/e2-1.json || return 1
  log "tp21-$M: TP=2 RoCE: $(grep -h 'RoCEnante all-reduce is live' $G/capturas/$M-r2-arranque.log | head -1 | sed 's/^.*\] //' | cut -c1-90)"
  mkdir -p $T/tp2; python3 $D/tools-check.py $R2 $T/tp2 > $T/tp2-tools.log 2>&1; log "tp21-$M: TP=2 tools exit=$? $(tail -1 $T/tp2-tools.log | cut -c1-100)"
  mide $R2 $T/tp2-decode 1,4,8,10,16; log "tp21-$M: TP=2 decode $(python3 $O/resumen-decode.py $T/tp2-decode.json)"
}
uno() {
  [ -s $G/capturas/E2-tp1-2.json ] || bash $G/prueba-gdn-mix.sh E2 1 || return 1
  bash $G/prueba-gdn-mix.sh $M 1 $G/capturas/E2-tp1-1.json || return 1
  log "tp21-$M: TP=1 $(grep -hE 'GPU KV cache size|Available KV cache memory' $G/capturas/$M-tp1-arranque.log | sed 's/^.*\] //' | cut -c1-80 | sort -u | paste -sd' ') · dgx3 $(ssh dgx3 free -g | awk '/Mem/{print "disponible "$7" de "$2" GiB"}')"
  mkdir -p $T/tp1; python3 $D/tools-check.py $R1 $T/tp1 > $T/tp1-tools.log 2>&1; log "tp21-$M: TP=1 tools exit=$? $(tail -1 $T/tp1-tools.log | cut -c1-100)"
  mide $R1 $T/tp1-decode 1,4,8; log "tp21-$M: TP=1 decode $(python3 $O/resumen-decode.py $T/tp1-decode.json)"
}
dos > $T/dos.log 2>&1 & p2=$!
ok1=0; [ -n "${SIN_UNO:-}" ] || { uno > $T/uno.log 2>&1 & p1=$!; }  # SIN_UNO=1: la 1x ya está arriba y medida
wait $p2; ok2=$?; [ -n "${SIN_UNO:-}" ] || { wait $p1; ok1=$?; }
grep -h 'tp21-A\|gdn-mix: compare' $T/dos.log $T/uno.log
[ $ok2 = 0 ] && [ $ok1 = 0 ] || { log "tp21-$M: sin los dos arriba no hay prueba conjunta (TP=2 $ok2, TP=1 $ok1; ver $T/dos.log y uno.log)"; exit 1; }
for cc in "4 4" "8 8" "16 8"; do  # los dos a la vez; 16 y 8 son los topes de seqs de la 2x y la 1x
  set -- $cc; mide $R2 $T/junto-c$1+$2-tp2 $1 & p=$!; mide $R1 $T/junto-c$1+$2-tp1 $2; wait $p
  log "tp21-$M: juntos c$1+c$2: TP=2 $(python3 $O/resumen-decode.py $T/junto-c$1+$2-tp2.json | cut -d' ' -f1-2) · TP=1 $(python3 $O/resumen-decode.py $T/junto-c$1+$2-tp1.json | cut -d' ' -f1-2)"
done
log "tp21-$M: hecho; arriba la 2x y la 1x con $M, residente a 0 y árbitro en pausa"
