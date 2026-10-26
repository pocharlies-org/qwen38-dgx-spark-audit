#!/usr/bin/env bash
# DGX-852 (Dani 14-10): su fork + estado SSM en BF16 (E2 de DGX-847: GDN triton en vez del b12x, que exige FP32) + vocabulario del
# borrador en castellano. TP=2 en dgx2+dgx1 (receta 2x) y TP=1 en dgx3 (receta 1x) a la vez, con sus imágenes y sus recetas tal
# cual (ursu.yaml) salvo eso. Cada uno por separado (tools + decode) y luego los dos a la vez. El residente queda a 0 y el árbitro
# en pausa; después va la prueba en TP=3.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
T=$D/mtp/definitivo/e2; mkdir -p $T
R2=http://spark2:8000; R1=http://spark3:8000
export MODEL=qwen3.8-flash-next
E2_ARGS='--mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel triton --additional-config {"gdn_prefill_backend":"triton","gdn_decode_kernel":"triton"}'

kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
for h in dgx dgx2 dgx3; do  # NVRM tarda en devolver la memoria unificada (espera-2x.sh: subir antes colgó el driver de dgx1)
  t0=$(date +%s)
  until [ "$(ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")" -ge 100 ]; do
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "e2: $h no recupera la memoria en 15 min"; exit 1; }; sleep 10
  done
done

kubectl create configmap -n llm ursu-recipe --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap -n llm ursu-env-2x --from-env-file=$O/recipe/env-2x.env --from-env-file=$O/recipe/env-sparkrun-2x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap -n llm ursu-env-1x --from-env-file=$O/recipe/env-1x.env --from-env-file=$O/recipe/env-sparkrun-1x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label cm -n llm ursu-recipe ursu-env-2x ursu-env-1x dgx-847=prueba --overwrite >/dev/null
# E2 (vuelta-residente.sh de DGX-847, la que arrancó): BF16 + GDN triton + checkpoints GDN diferidos fuera; vocabulario castellano
E2_ARGS="$E2_ARGS" python3 - $O/ursu.yaml <<'PY' | kubectl apply -f - >/dev/null
import os, sys, yaml
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
for d in docs:
    s = d["spec"]["template"]["spec"]; c = s["containers"][0]
    s["volumes"].append({"name": "mtp-vocab-es", "configMap": {"name": "qwen38fn-mtp-draft-vocab"}})
    c["volumeMounts"].append({"name": "mtp-vocab-es", "mountPath": "/opt/mtp-vocab-es", "readOnly": True})
    c.setdefault("env", []).extend([
        {"name": "EXTRA_ARGS", "value": os.environ["E2_ARGS"]},
        {"name": "VLLM_GDN_DEFERRED_CHECKPOINTS", "value": "0"},
        {"name": "VLLM_GDN_COMPACT_RECORDS", "value": "0"},  # la 1x lo trae a 1 y exige el GDN b12x (14-10 06:49Z)
        {"name": "VLLM_MTP_DRAFT_VOCAB", "value": "/opt/mtp-vocab-es/ids-es-K131072.txt.gz"}])
print(yaml.safe_dump_all(docs))
PY
log "e2: su fork + BF16 + vocab castellano: TP=2 (dgx2+dgx1) y TP=1 (dgx3) arrancando"

arriba() {  # $1 URL, $2 selector, $3 nombre; al primer reinicio se da por fallida (un fallo de configuración se repite)
  local t0=$(date +%s)
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $1/health)" = 200 ]; do
    kubectl get pods -n llm -l "$2" -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | grep -qE '[1-9]' \
      && { log "e2: $3 se ha reiniciado"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "e2: $3 sin health en 60 min"; return 1; }
    sleep 15
  done; log "e2: $3 arriba en $(( ($(date +%s) - t0) / 60 )) min"
}
arriba $R2 'app in (ursu-2x-head,ursu-2x-worker)' TP=2 & a2=$!
arriba $R1 'app=ursu-1x' TP=1 & a1=$!
wait $a2; ok2=$?; wait $a1; ok1=$?
kubectl logs -n llm deploy/ursu-2x-head > $T/tp2-arranque.log 2>&1
kubectl logs -n llm deploy/ursu-1x > $T/tp1-arranque.log 2>&1
for n in tp2 tp1; do
  log "e2: $n $(grep -hE 'attention block size|Model loading took|Available KV cache memory|GPU KV cache size|MTP_DRAFT_VOCAB|draft vocab' $T/$n-arranque.log | sed 's/^.*\] //' | cut -c1-100 | sort -u | paste -sd' ')"
done

mide() {  # $1 URL, $2 fichero sin extensión, $3 concurrencias
  python3 $TOOLS/qwen38_decode_sweep.py --base-url $1 --model $MODEL --concurrency $3 --warmup 20 --duration 60 --reps 1 \
    --max-tokens 256 --prompts $TOOLS/fixtures/qwen38_ab/prompts.jsonl --timeout 600 --sample-interval 5.0 --long-workers 0 --long-tokens 32000 \
    --max-model-len 262144 --seed 496368 --out $2.json > $2.log 2>&1
}
[ $ok2 = 0 ] && { mkdir -p $T/tp2; python3 $D/tools-check.py $R2 $T/tp2 > $T/tp2-tools.log 2>&1; log "e2: TP=2 tools exit=$? $(tail -1 $T/tp2-tools.log | cut -c1-100)"
  mide $R2 $T/tp2-decode 1,4,8,10; log "e2: TP=2 decode $(python3 $O/resumen-decode.py $T/tp2-decode.json)"; }
[ $ok1 = 0 ] && { mkdir -p $T/tp1; python3 $D/tools-check.py $R1 $T/tp1 > $T/tp1-tools.log 2>&1; log "e2: TP=1 tools exit=$? $(tail -1 $T/tp1-tools.log | cut -c1-100)"
  mide $R1 $T/tp1-decode 1,4,8; log "e2: TP=1 decode $(python3 $O/resumen-decode.py $T/tp1-decode.json)"; }
[ $ok2 = 0 ] && [ $ok1 = 0 ] || { log "e2: sin los dos arriba no hay prueba conjunta"; exit 1; }
for c in 4 8; do  # los dos a la vez, mismo nivel en cada uno (total 2c); la 1x tiene 8 seqs
  mide $R2 $T/junto-c$c-tp2 $c & p=$!; mide $R1 $T/junto-c$c-tp1 $c; wait $p
  log "e2: juntos c$c+c$c: TP=2 $(python3 $O/resumen-decode.py $T/junto-c$c-tp2.json | cut -d' ' -f1-2) · TP=1 $(python3 $O/resumen-decode.py $T/junto-c$c-tp1.json | cut -d' ' -f1-2)"
done
log "e2: hecho; arriba su 2x y su 1x (E2), residente a 0 y árbitro en pausa"
