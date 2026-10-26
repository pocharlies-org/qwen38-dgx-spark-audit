#!/usr/bin/env bash
# DGX-912: puerta -> merge -> builds backend+UI del SHA -> rollout -> nombres en el /api/activity vivo.
set -u
export KUBECONFIG=~/.kube/config
R=pocharlies-org/dgx-infra; OPS=/home/user/bench-results/dgx-847/ops
out=$($OPS/espera-lista.sh $R 1188 3600); rc=$?; echo "$out"
[ $rc -eq 0 ] || { echo "FALLO puerta rc=$rc"; exit $rc; }
sha=$(gh pr view 1188 -R $R --json mergeCommit -q .mergeCommit.oid); echo "$(date -u +%T) merge $sha"
fin=$((SECONDS + 9000))
for wf in build-dashboard-backend.yml build-dashboard-ui.yml; do
  while :; do
    (( SECONDS < fin )) || { echo "TOPE esperando $wf"; exit 4; }
    st=$(gh run list -R $R --workflow $wf --commit $sha --json status,conclusion -q '.[0] | (.status+" "+(.conclusion//""))' 2>/dev/null)
    case "$st" in
      "completed success") echo "$(date -u +%T) $wf OK"; break;;
      completed*) echo "$(date -u +%T) $wf FALLÓ: $st"; exit 1;;
    esac
    sleep 60
  done
done
short=${sha:0:12}
while :; do
  (( SECONDS < fin )) || { echo "TOPE esperando rollout"; exit 4; }
  img=$(kubectl get deploy -n control-nexus dgx-dashboard-ui -o jsonpath='{..image}')
  names=$(curl -s -m 20 http://10.43.80.147:9002/api/activity | python3 -c '
import sys,json
v=json.load(sys.stdin).get("vllm",[])
print(";".join(sorted({s.get("name","") for s in v if "qwen38" in s.get("id","")})))' 2>/dev/null)
  if grep -q "$short" <<<"$img" && [ "$names" = "qwen38-flash-next principal;qwen38-flash-next secundario" ]; then
    echo "$(date -u +%T) EN VIVO: ui=$img names=$names"; exit 0
  fi
  sleep 60
done
