#!/usr/bin/env bash
# DGX-847 prueba C: perfil torch de su 2× (receta tal cual, sin la PLE en disco de G) sirviendo 64K en caché + 2048 nuevos
# a c1 y c8. profiler_config entra en el hash de compilación: el arranque recompila. Al salir suelta el cierre de la ventana.
set -u
export KUBECONFIG=/home/user/.kube/config
D=/home/user/bench-results/dgx-847; R=http://spark2:8000; T=$D/pruebas/c-perfil; mkdir -p $T
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
# el cierre espera también a las pruebas del 1× (pruebas-1x.sh, en paralelo en dgx3)
trap 'log "C: fin"; until grep -q "1x: fin" $D/pruebas/1x.log 2>/dev/null; do sleep 30; done; rm -f $D/ops/NO-CERRAR; log "NO-CERRAR fuera: cierre de la ventana"' EXIT
until grep -q 'G: fin' $D/pruebas/g.log 2>/dev/null; do sleep 30; done
for h in dgx dgx2; do ssh $h 'sudo -n rm -rf /srv/ursu-cache/2x/perfil-c'; done
. $D/ops/espera-2x.sh; reinicia_2x VLLM_PLE_MMAP- VLLM_PLE_MMAP_WILLNEED_MAX- VLLM_PLE_MMAP_KEEPALIVE_MS- VLLM_PLE_MMAP_PREFILL_WILLNEED- \
  EXTRA_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/xdg/perfil-c --profiler-config.torch_profiler_with_stack=false" || exit 1
log "C: reinicio de la 2× con el perfilador (recompila)"; t0=$(date +%s)
espera_2x || { kubectl logs -n llm deploy/ursu-2x-head --tail=200 > $T/arranque.log 2>&1; exit 1; }
log "2× lista en $(( ($(date +%s) - t0) / 60 )) min"
kubectl logs -n llm deploy/ursu-2x-head > $T/arranque.log 2>&1; grep -iE 'profil|attention block size' $T/arranque.log | cut -c1-200
python3 -I $D/prueba-c.py $R $T/carga.json; log "carga exit=$?"
sleep 60
for h in dgx2:head dgx:worker; do ssh ${h%%:*} 'sudo -n ls -la /srv/ursu-cache/2x/perfil-c/' | sed "s/^/${h#*:}: /"; mkdir -p $T/${h#*:}; ssh ${h%%:*} 'sudo -n tar -C /srv/ursu-cache/2x/perfil-c -cf - .' | tar -C $T/${h#*:} -xf -; done  # las escribe root
python3 -I $D/perfil-resumen.py $(find $T/head $T/worker -name "*.json*" | sort) > $T/resumen.txt 2>&1; log "resumen exit=$?"
cat $T/resumen.txt
