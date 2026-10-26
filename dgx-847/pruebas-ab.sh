#!/usr/bin/env bash
# DGX-847: A (caché con contexto de 65535 exactos) y luego B (max-num-batched-tokens 4096), en la 2×, cuando acabe su calidad.
D=/home/user/bench-results/dgx-847
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
until grep -q '2x: calidad exit=' $D/chain-2x.log; do sleep 30; done
log "calidad 2× acabada: $(grep '2x: calidad exit=' $D/chain-2x.log | tail -1)"
sleep 20
log "A: inicio (contexto 65535)"
python3 -I $D/prueba-a-cache.py http://spark2:8000 $D/pruebas/a-cache-2x-ctx65535.json 65535 | sed "s/^/A: /"
log "A: fin"
log "B: inicio"
$D/prueba-b.sh 2>&1 | sed "s/^/B: /"
log "B: fin"
