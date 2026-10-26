#!/usr/bin/env bash
# DGX-852 (Dani 14-10): TP=2 en dgx1+dgx2 y TP=1 en dgx3 a la vez, con la vista tp12 (ursu original, PLE en FP8, sin relleno),
# SU sistema (Dani 14-10): MoE b12x sin EP con el micro de ursu, MTP b12x (w31), k=4, vocabulario recortado; de lo nuestro, el SSM BF16. Sin KV fijo (KV_CACHE_MEMORY_BYTES=0): vLLM perfila con GPU_MEM_UTIL
# y su log da el KV máximo de cada uno. Cada uno por separado (herramientas + decode) y luego los dos a la vez, nivel a nivel
# (el reparto lo hace el cliente, como lo haría LiteLLM con dos réplicas). Árbitro en pausa todo el rato; no vuelve a TP=3.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
V=/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp12; REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
DV=/opt/qwen38fn-mtp-dv/ids-es-K131072.txt.gz; K=${K:-4}; T=$D/mtp/definitivo/tp21-ursu; mkdir -p $T
TP1=http://spark3:8000
export MODEL=qwen38-flash-next
[ -f $T/antes-tp3.json ] || kubectl get deploy -n llm $DEPS -o json > $T/antes-tp3.json  # la vuelta a TP=3 sale de aquí
kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done

pon() {  # $1 deploy, $2 TP, $3 rango, $4 dir PLE del nodo
  kubectl patch deploy -n llm $1 --type strategic \
    -p "{\"spec\":{\"template\":{\"spec\":{\"volumes\":[{\"name\":\"ple-table\",\"hostPath\":{\"path\":\"$4\",\"type\":\"DirectoryOrCreate\"}}]}}}}" >/dev/null
  kubectl set env -n llm deploy/$1 -c ple-table CKPT_DIR=$V CKPT_REVISION=$REV TP_SIZE=$2 TP_RANK=$3 >/dev/null
  kubectl set env -n llm deploy/$1 -c vllm QWEN_TP=$2 VLLM_NODE_RANK=$3 MODEL_PATH=$V MODEL_REVISION=$REV KV_CACHE_MEMORY_BYTES=0 \
    QWEN38_B12X=1 QWEN38_B12X_MOE=1 QWEN38_B12X_URSU=1 QWEN38_B12X_MTP31=1 QWEN38_B12X_MOE_WA- VLLM_MTP_DRAFT_VOCAB=$DV \
    "SPECULATIVE_CONFIG=$(spec_k $K)" >/dev/null
}
pon qwen38-flash-next-head 2 0 /home/user/.cache/qwen38fn-ple-ursu-tp2
pon qwen38-flash-next-worker 2 1 /home/user/.cache/qwen38fn-ple-ursu-tp2
pon qwen38-flash-next-worker-dgx3 1 0 /home/user/.cache/qwen38fn-ple-ursu-tp1
# TP=1: 72,3 GiB de pesos; con 0.73 el KV salía en -1,1 GiB (14-10 06:12Z). 0.88 de 121 GiB deja ~14 GiB al sistema y al page cache
kubectl set env -n llm deploy/qwen38-flash-next-worker-dgx3 -c vllm GPU_MEM_UTIL=0.88 >/dev/null
# TP=2 solo por el puerto 0 dgx1<->dgx2 (ring.conf: filas tp, mitades A y B); el generador del anillo es del TP=3
for d in qwen38-flash-next-head qwen38-flash-next-worker; do
  kubectl set env -n llm deploy/$d -c vllm QWEN_NCCL_RING_HOST- NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 >/dev/null
done
kubectl scale deploy -n llm $DEPS --replicas=1 >/dev/null; log "tp21: TP=2 (dgx1+dgx2) y TP=1 (dgx3) con su sistema (MoE b12x sin EP, k=$K) arrancando, KV por perfilado"

arriba() {  # $1 URL, $2 patrón de pods, $3 nombre; 1 si el head (o el TP=1) reinicia o un worker reinicia 2 veces
  local t0=$(date +%s)
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $1/health)" = 200 ]; do
    kubectl get pods -n llm --no-headers | awk -v p="$2" '$0 ~ p && $3 != "Terminating" && ($4 >= 2 || ($1 !~ /-worker-[0-9a-f]+-/ && $4 >= 1)) {f=1} END{exit !f}' \
      && { log "tp21: $3 se ha reiniciado"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "tp21: $3 sin health en 90 min"; return 1; }
    sleep 20
  done; log "tp21: $3 arriba en $(( ($(date +%s) - t0) / 60 )) min"
}
arriba $RES '^qwen38-flash-next-(head|worker-[0-9a-f]+-)' TP=2 & a2=$!
arriba $TP1 '^qwen38-flash-next-worker-dgx3' TP=1 & a1=$!
wait $a2; ok2=$?; wait $a1; ok1=$?
kubectl logs -n llm deploy/qwen38-flash-next-head -c vllm > $T/tp2-arranque.log 2>&1
kubectl logs -n llm deploy/qwen38-flash-next-worker-dgx3 -c vllm > $T/tp1-arranque.log 2>&1
for n in tp2 tp1; do
  log "tp21: $n $(grep -hE 'Model loading took|Available KV cache memory|GPU KV cache size' $T/$n-arranque.log | sed 's/^.*\] //' | cut -c1-110 | paste -sd' ' )"
done

mide() {  # $1 URL, $2 dir, $3 concurrencias
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $1 --model qwen38-flash-next --concurrency $3 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $2.json > $2.log 2>&1
}
[ $ok2 = 0 ] && { mkdir -p $T/tp2; python3 $D/tools-check.py $RES $T/tp2 > $T/tp2-tools.log 2>&1; log "tp21: TP=2 tools exit=$? $(tail -1 $T/tp2-tools.log | cut -c1-100)"
  mide $RES $T/tp2-decode 1,4,8,10; log "tp21: TP=2 decode $(python3 $O/resumen-decode.py $T/tp2-decode.json)"; }
[ $ok1 = 0 ] && { mkdir -p $T/tp1; python3 $D/tools-check.py $TP1 $T/tp1 > $T/tp1-tools.log 2>&1; log "tp21: TP=1 tools exit=$? $(tail -1 $T/tp1-tools.log | cut -c1-100)"
  mide $TP1 $T/tp1-decode 1,4,8,10; log "tp21: TP=1 decode $(python3 $O/resumen-decode.py $T/tp1-decode.json)"; }
[ $ok2 = 0 ] && [ $ok1 = 0 ] || { log "tp21: sin los dos arriba no hay prueba conjunta"; exit 1; }
for c in 4 8 10; do  # los dos a la vez, mismo nivel en cada uno (total 2c)
  mide $RES $T/junto-c$c-tp2 $c & p=$!; mide $TP1 $T/junto-c$c-tp1 $c; wait $p
  log "tp21: juntos c$c+c$c: TP=2 $(python3 $O/resumen-decode.py $T/junto-c$c-tp2.json | cut -d' ' -f1-2) · TP=1 $(python3 $O/resumen-decode.py $T/junto-c$c-tp1.json | cut -d' ' -f1-2)"
done
log "tp21: hecho; residente en TP=2 + TP=1 con el árbitro en pausa (vuelta a TP=3 desde $T/antes-tp3.json)"
