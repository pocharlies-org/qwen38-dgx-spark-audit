#!/usr/bin/env bash
# DGX-847 F: nuestras palancas en su 2×. El estado SSM en BF16 es imposible con sus kernels b12x (prueba E), así que F mide
# las otras dos: 4096 tokens por paso y 2048 por petición (--long-prefill-token-threshold). Reinicio, prueba A (65535) y benchy 64K.
D=/home/user/bench-results/dgx-847
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
TAG=f-nuestros-3; EX="--max-num-batched-tokens 4096 --long-prefill-token-threshold 2048"
log "$TAG: inicio ($EX; sin BF16, imposible con b12x)"
TAG=$TAG EXTRA="$EX" $D/prueba-b.sh 2>&1 | sed -u "s/^/$TAG: /"
if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://spark2:8000/health)" = 200 ]; then
  python3 -I $D/prueba-a-cache.py http://spark2:8000 $D/pruebas/$TAG/a-cache-ctx65535.json 65535 | sed -u "s/^/$TAG A: /"
fi
log "$TAG: fin"
