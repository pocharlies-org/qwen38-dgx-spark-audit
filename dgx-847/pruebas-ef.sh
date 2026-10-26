#!/usr/bin/env bash
# DGX-847: E = su 2× con el estado SSM en BF16 (nuestra palanca, ablación de Jordi 27-09); F = E + nuestros dos topes del
# planificador (4096 por paso, 2048 por petición). Cada una: reinicio, tamaño de bloque del log, prueba A (65535) y benchy 64K.
D=/home/user/bench-results/dgx-847
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
until grep -q '^.*B: fin' $D/pruebas/ab.log 2>/dev/null; do sleep 30; done
for t in "e-ssm-bf16|--mamba-ssm-cache-dtype bfloat16" "f-nuestros-3|--mamba-ssm-cache-dtype bfloat16 --max-num-batched-tokens 4096 --long-prefill-token-threshold 2048"; do
  TAG=${t%%|*}; EX=${t#*|}
  log "$TAG: inicio ($EX)"
  TAG=$TAG EXTRA="$EX" $D/prueba-b.sh 2>&1 | sed "s/^/$TAG: /"
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark2:8000/health)" = 200 ]; then
    python3 -I $D/prueba-a-cache.py http://spark2:8000 $D/pruebas/$TAG/a-cache-ctx65535.json 65535 | sed "s/^/$TAG A: /"
  fi
  log "$TAG: fin"
done
