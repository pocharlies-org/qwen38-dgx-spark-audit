#!/usr/bin/env bash
# DGX-852: prueba de red de su all-reduce RoCE en el anillo de 3 Sparks (test_roce_ring.py en los tres a la vez), antes de
# servir con él. Un pod por rango con la plantilla de ursu-3x.yaml tal cual (red del host, dispositivos, env), otro comando.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
R=$D/mtp/definitivo/fork-tp3/roce-anillo
F="deploy/ursu-fork-3x-head deploy/ursu-fork-3x-worker-dgx1 deploy/ursu-fork-3x-worker-dgx3"
kubectl scale -n llm $F --replicas=0 >/dev/null; while kubectl get pods -n llm -l dgx-852=fork-tp3 --no-headers 2>/dev/null | grep -q .; do sleep 5; done
kubectl create configmap -n llm roce-anillo --from-file=$R/patch_roce_ring.py --from-file=$R/roce_ring_map.py \
  --from-file=$R/ring.conf --from-file=$R/test_roce_ring.py --dry-run=client -o yaml | kubectl apply -f - >/dev/null
for d in ursu-fork-3x-head ursu-fork-3x-worker-dgx1 ursu-fork-3x-worker-dgx3; do
  kubectl get deploy -n llm $d -o json | python3 -c '
import json, sys
t = json.load(sys.stdin)["spec"]["template"]; s = t["spec"]; c = s["containers"][0]
for k in ("args", "readinessProbe", "livenessProbe", "startupProbe"): c.pop(k, None)
c["command"] = ["bash", "-c", "set -e; B12X_ROCE_RING=1 python3 -I /tmp/roce-anillo/patch_roce_ring.py; "
  "set -a; eval \"$(python3 -I /tmp/roce-anillo/roce_ring_map.py --host $QWEN_NCCL_RING_HOST)\"; set +a; "
  "export MASTER_ADDR=spark2 MASTER_PORT=25001 GLOO_SOCKET_IFNAME=enP7s7 B12X_ROCE_SPIN_LIMIT=30000000; "
  "python3 -I /tmp/roce-anillo/test_roce_ring.py"]
c.setdefault("volumeMounts", []).append({"name": "roce-anillo", "mountPath": "/tmp/roce-anillo", "readOnly": True})
s.setdefault("volumes", []).append({"name": "roce-anillo", "configMap": {"name": "roce-anillo"}})
s["restartPolicy"] = "Never"
name = "roce-red-" + sys.argv[1].split("-")[-1]
print(json.dumps({"apiVersion": "v1", "kind": "Pod", "spec": s,
  "metadata": {"name": name, "namespace": "llm", "labels": {"dgx-852": "roce-red"}}}))' $d | kubectl apply -f - >/dev/null
done
log "roce-red: tres pods lanzados"
t0=$(date +%s)
until [ "$(kubectl get pods -n llm -l dgx-852=roce-red -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}' | grep -cE 'Succeeded|Failed')" = 3 ]; do
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "roce-red: sin terminar en 15 min"; break; }; sleep 10
done
for p in $(kubectl get pods -n llm -l dgx-852=roce-red -o jsonpath='{.items[*].metadata.name}'); do
  kubectl logs -n llm $p > $R/$p.log 2>&1; log "roce-red: $p $(kubectl get pod -n llm $p -o jsonpath='{.status.phase}') · $(tail -2 $R/$p.log | tr '\n' ' ' | cut -c1-200)"
done
kubectl delete pods -n llm -l dgx-852=roce-red --wait=false >/dev/null; kubectl delete cm -n llm roce-anillo >/dev/null
