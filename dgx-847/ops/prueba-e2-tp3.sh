#!/usr/bin/env bash
# DGX-852 (Dani 14-10): su fork en TP=3 (dgx2 head + dgx1 + dgx3) con E2 (SSM BF16, GDN triton) y vocabulario castellano;
# vista y receta de mtp/definitivo/fork-tp3 (INFORME.md). Baja la 2x y la 1x, arranque frío, tools, greedy contra E2 TP=2 y decode.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
T=$D/mtp/definitivo/fork-tp3; S=$T${QWEN_GDN_MIX:+/$QWEN_GDN_MIX}${ROCE_RING:+-roce}; mkdir -p $S; G=$D/mtp/definitivo/gdn-mix/capturas; R=http://spark2:8000; export MODEL=qwen3.8-flash-next
if [ -n "${SIGUE:-}" ]; then  # ya aplicado: ConfigMaps al día y los tres rangos a 0 y a 1 juntos (cita del TCPStore)
  kubectl create configmap -n llm ursu-fork-env-3x --from-env-file=$T/env-3x.env --from-env-file=$T/env-ring-3x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  F="deploy/ursu-fork-3x-head deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3"
  kubectl scale -n llm $F --replicas=0 >/dev/null; while kubectl get pods -n llm -l dgx-852=fork-tp3 --no-headers | grep -q .; do sleep 5; done
  kubectl scale -n llm $F --replicas=1 >/dev/null; log "tp3: rangos rehechos (SIGUE)"
else
kubectl scale -n llm deploy/ursu-2x-head deploy/ursu-2x-worker deploy/ursu-1x --replicas=0 >/dev/null
# un TP=3 anterior también baja entero (los tres rangos a la vez); aplicar-3x.sh los sube juntos
kubectl scale -n llm deploy/ursu-fork-3x-head deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3 --replicas=0 >/dev/null 2>&1
while kubectl get pods -n llm --no-headers | grep -q '^ursu-'; do sleep 5; done
for h in dgx dgx2 dgx3; do t0=$(date +%s)
  until [ "$(ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")" -ge 100 ]; do
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "tp3: $h no recupera la memoria"; exit 1; }; sleep 10; done
done
# con el parche GDN el head del dgx2 convive con stt-turbo (residente de llm-tp ahí): 0.8 no cabe con B2 (14-10 12:31Z)
EXTRA=${QWEN_GDN_MIX:+--gpu-memory-utilization ${UTIL3:-0.70}} bash $T/aplicar-3x.sh || { log "tp3: aplicar-3x.sh falla"; exit 1; }
fi
t0=$(date +%s)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R/health)" = 200 ]; do
  kubectl get pods -n llm -l dgx-852=fork-tp3 -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | grep -qE '[1-9]' \
    && { log "tp3: un rango se ha reiniciado"; kubectl logs -n llm deploy/ursu-fork-3x-head --tail=-1 > $S/arranque-fallo.log 2>&1; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "tp3: sin health en 90 min"; exit 1; }
  sleep 20
done
log "tp3: arriba en $(( ($(date +%s) - t0) / 60 )) min"
kubectl logs -n llm deploy/ursu-fork-3x-head --tail=-1 > $S/arranque.log 2>&1
log "tp3: $(grep -hE 'attention block size|Model loading took|Available KV cache memory|GPU KV cache size|draft vocab|GDN decode kernel' $S/arranque.log | sed 's/^.*\] //' | cut -c1-100 | sort -u | paste -sd' ')"
mkdir -p $S/tools; python3 $D/tools-check.py $R $S/tools > $S/tools.log 2>&1; log "tp3: tools exit=$? $(tail -1 $S/tools.log | cut -c1-100)"
python3 -I $D/mtp/definitivo/gdn-mix/greedy_compare.py --url $R --model $MODEL --compare $G/e2-1.json --noise $G/e2-2.json \
  --out $S/tp3-vs-e2tp2.json > $S/tp3-vs-e2tp2.log 2>&1; log "tp3: greedy contra E2 TP=2 exit=$? $(tail -1 $S/tp3-vs-e2tp2.log | cut -c1-120)"
python3 $TOOLS/qwen38_decode_sweep.py --base-url $R --model $MODEL --concurrency 1,4,8,10,16 --warmup 20 --duration 60 --reps 1 \
  --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
  --max-model-len 262144 --seed 496368 --out $S/decode.json > $S/decode.log 2>&1
log "tp3: decode $(python3 $O/resumen-decode.py $S/decode.json)"
log "tp3: hecho (su fork TP=3 arriba; 2x y 1x a 0)"
