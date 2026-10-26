#!/usr/bin/env bash
# DGX-852 (Dani 14-10): prefill en frío de ursu mixto (A) en 2 Sparks y en 1 Spark con su checkpoint original (PLE NVFP4),
# medido igual que en 3 Sparks (ops/prefill.sh). Baja el TP=3 entero; los dos a la vez, cada uno en su hardware.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
G=$D/mtp/definitivo/gdn-mix; P=$D/mtp/definitivo/prefill
kubectl scale -n llm deploy/ursu-fork-3x-head deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3 --replicas=0 >/dev/null
while kubectl get pods -n llm -l dgx-852=fork-tp3 --no-headers 2>/dev/null | grep -q .; do sleep 5; done
( N=A-pf2 EXTRA="--gpu-memory-utilization 0.75" bash $G/prueba-gdn-mix.sh A 2 && bash $O/prefill.sh http://spark2:8000 $P/mixto-tp2 ) > $P/tp2.log 2>&1 & a=$!
( N=A-pf1 bash $G/prueba-gdn-mix.sh A 1 && bash $O/prefill.sh http://spark3:8000 $P/mixto-tp1 ) > $P/tp1.log 2>&1 & b=$!
wait $a; ea=$?; wait $b; eb=$?
log "prefill-tp21: TP=2 exit=$ea · TP=1 exit=$eb"
