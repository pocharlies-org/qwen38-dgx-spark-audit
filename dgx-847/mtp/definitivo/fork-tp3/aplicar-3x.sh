#!/usr/bin/env bash
# DGX-852: lanza el TP=3 del fork de ursuciprian (E2 + vocabulario del borrador en castellano).
# Copia de tp3/aplicar-3x.sh (el original de DGX-847 no se toca) con dos diferencias:
#  - NO hay paso `--publicar`: las dos vistas se construyen EN dgx1 (construir-vistas.sh) y dgx2 y dgx3
#    las ven por los PVC ro de NFS que ya usan los residentes TP=3 (hf-cache-nvidia-dgx-ro y
#    hf-cache-dgx1-dgx3-llm-ro). Nadie copia 133 GB a ningún sitio.
#  - los ConfigMaps y los despliegues llevan nombre propio (ursu-fork-*) para no pisar los de DGX-847.
# Da por hecho la ventana YA abierta por ops/ventana.sh (ArgoCD root y ai sin auto-sync, Job
# llm-deploy-pause, qwen38 a 0, drop_caches). La vuelta es la de siempre: ops/cierre.sh.
set -euo pipefail
export KUBECONFIG=/home/user/.kube/config
T=/home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3
log() { echo "$(date -u +%H:%M:%SZ) $*"; }

log "compruebo las dos vistas en dgx1 (las sirve el NFS ro a dgx2 y dgx3)"
C=/home/dibanerz/.cache/huggingface
# la vista del modelo lleva config; la copia MXFP8 NO (lo pone la vista: --mxfp8-copy no escribe config)
VISTA=${VISTA:-qwen38fn-mtp-ursu-universal}  # la que monta serve-3x.sh por defecto (PLE NVFP4); la FP8 era -nvfp4-tp3-fork
for d in $VISTA qwen38fn-mtp-ursu-mxfp8-tp3; do
  t="test -f $C/$d/model.safetensors.index.json"
  [ "$d" = "$VISTA" ] && t="$t -a -f $C/$d/config.json"
  ssh dgx "$t" \
    || { log "FALTA $d en dgx1 — construye:  ssh dgx bash /home/dibanerz/dgx-852/construir-vistas.sh"; exit 1; }
  log "  dgx1:$d ok"
done

log "compruebo lo que el manifiesto monta"
for pvc in hf-cache-nvidia-dgx-ro hf-cache-dgx1-dgx3-llm-ro; do
  ph=$(kubectl -n llm get pvc $pvc -o jsonpath='{.status.phase}')
  [ "$ph" = Bound ] || { log "PVC $pvc en $ph, no Bound"; exit 1; }
  log "  pvc $pvc Bound"
done
kubectl -n llm get cm qwen38fn-mtp-draft-vocab -o json | python3 -c '
import json, sys
d = json.load(sys.stdin)
k = "ids-es-K131072.txt.gz"
if k not in (d.get("binaryData") or {}) and k not in (d.get("data") or {}):
    sys.exit(f"el ConfigMap qwen38fn-mtp-draft-vocab no trae {k}")
print("  qwen38fn-mtp-draft-vocab trae", k)'

log "compruebo los tres nodos: sin DiskPressure y con las dos imagenes que hacen falta"
IMG=ghcr.io/ursuciprian/spark-vllm-b12x@sha256:8abb60eeeb79ca13680341c17d87975801da1218c06df2505856b23da7ed1821
RING=internal.example/homelab/nccl-ring@sha256:79db26e7877a099847cf12c2067a3dd3f02156c51a0d77386691b272316ce77e
for h in dgx2 dgx dgx3; do
  case $h in dgx) node=nvidia-dgx;; dgx2) node=gx10-ec3d;; *) node=dgx3;; esac
  if kubectl get node $node -o jsonpath='{.spec.taints}' | grep -q disk-pressure; then
    log "  $h: TAINT disk-pressure (el kubelet desaloja lo que se programe). No lanzar: libera disco o espera."
    exit 1
  fi
  for img in "$IMG" "$RING"; do
    ssh $h "sudo -n k3s ctr images ls -q" 2>/dev/null | grep -qx "$img" \
      || log "  AVISO $h: no tiene en cache $(basename $img | cut -c1-40)… (harbor no tiene credenciales en el nodo: el pull del volumen nccl-ring FALLA; hace falta el .so por otro camino)"
  done
  log "  $h sin disk-pressure"
done

log "ConfigMaps de la 3x del fork"
G=/home/user/bench-results/dgx-847/mtp/definitivo/gdn-mix  # el parche GDN (QWEN_GDN_MIX) va en la misma receta
kubectl create configmap -n llm ursu-fork-recipe-3x --from-file=$T/serve-3x.sh --from-file=$G/serve-gdn-mix.sh --from-file=$G/patch_gdn_mix.py \
  --from-file=$T/roce-anillo/patch_roce_ring.py --from-file=$T/roce-anillo/roce_ring_map.py --from-file=$T/roce-anillo/ring.conf \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap -n llm ursu-fork-env-3x --from-env-file=$T/env-3x.env --from-env-file=$T/env-ring-3x.env --dry-run=client -o yaml | kubectl apply -f -
kubectl label cm -n llm ursu-fork-recipe-3x ursu-fork-env-3x dgx-852=fork-tp3 --overwrite

log "tres rangos arriba"
# QWEN_GDN_MIX=A|B2: entrada serve-gdn-mix.sh (parche + flags GDN) y EXTRA_ARGS; vacío = la receta E2 tal cual
QWEN_GDN_MIX=${QWEN_GDN_MIX:-} EXTRA=${EXTRA:-} ROCE_RING=${ROCE_RING:-} python3 - $T/ursu-3x.yaml <<'PY' | kubectl apply -f -
import os, sys, yaml
mix, extra = os.environ["QWEN_GDN_MIX"], os.environ["EXTRA"]
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
for d in docs:
    if mix and d.get("kind") == "Deployment":
        c = d["spec"]["template"]["spec"]["containers"][0]
        c["args"] = ["bash", "/etc/ursu/serve-gdn-mix.sh"]
        c["env"] += [{"name": "QWEN_GDN_MIX", "value": mix}, {"name": "GDN_TP", "value": "3"}] \
            + ([{"name": "EXTRA_ARGS", "value": extra}] if extra else [])
    if os.environ["ROCE_RING"] == "1" and d.get("kind") == "Deployment":  # su RoCE en el anillo (serve-3x.sh)
        d["spec"]["template"]["spec"]["containers"][0]["env"].append({"name": "B12X_ROCE_RING", "value": "1"})
print(yaml.safe_dump_all(docs))
PY
kubectl -n llm rollout status deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3 --timeout=10s || true
log "TP=3 lanzado. El primer arranque es autotune FRIO (la semilla de la imagen esta claveada a TP=2 y"
log "  a la ruta del modelo), y el arranque del residente TP=3 ronda la media hora: seguir con"
log "    kubectl logs -n llm -l dgx-852=fork-tp3 -f   hasta 'b12x ready' + /health 200 en dgx2:8000"
