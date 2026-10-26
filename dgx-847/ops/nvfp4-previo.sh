#!/usr/bin/env bash
# DGX-852, antes de la ventana y con el residente sirviendo: la vista NVFP4 de ursu (640, la de EP) a dgx1, la tabla PLE de
# cada rango en su NVMe, la rueda b12x en el wheelhouse del NFS y la vista e768 (expertos rellenos para el MoE b12x), con su
# YAML (enlaza en dgx1 los 19 ficheros que comparte con la 640). La tabla PLE es la misma para las dos vistas.
# Rangos (ready.json de la tabla viva): gx10-ec3d = 0 (head), nvidia-dgx = 1 (worker), dgx3 = 2. El YAML del conversor los
# tenía cruzados (nvidia-dgx = 0); aquí manda el residente.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
C=$D/mtp/checkpoint; REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
NAME=qwen38fn-mtp-ursu-nvfp4-tp3
tolera() {  # los Sparks llevan el taint dedicated=llm:NoSchedule y los YAML del conversor no lo toleran (el 640 se quedó 30 min en Pending)
  python3 -c '
import sys, yaml
docs = [d for d in yaml.safe_load_all(sys.stdin) if d]
for d in docs:
    s = d.get("spec", {}).get("template", {}).get("spec")
    if s and s.get("nodeSelector", {}).get("kubernetes.io/hostname") in ("nvidia-dgx", "gx10-ec3d", "dgx3"):
        s["tolerations"] = [{"key": "dedicated", "operator": "Equal", "value": "llm", "effect": "NoSchedule"}]
print(yaml.safe_dump_all(docs))'
}

espera_job() {  # $1 = Job; 0 si acaba bien, 1 si falla o pasan $2 s (kubectl wait --for=Complete no vuelve si el Job falla)
  local t0=$(date +%s) st
  while [ $(( $(date +%s) - t0 )) -lt ${2:-7200} ]; do
    st=$(kubectl get job -n llm $1 -o jsonpath='{.status.succeeded}/{.status.failed}')
    case $st in 1/*) return 0 ;; */[1-9]*) return 1 ;; esac
    sleep 15
  done; return 1
}

publica() {  # serve (x86) + publish (dgx1) de la vista 640, del YAML del conversor
  python3 - $C/publicar-ursu-tp3.yaml $NAME <<'PY' | tolera | kubectl apply -f - >/dev/null
import sys, yaml
src, name = sys.argv[1], sys.argv[2]
docs = [d for d in yaml.safe_load_all(open(src)) if d and d["kind"] in ("Deployment", "Service", "Job") and "ple-table" not in d["metadata"]["name"]]
for d in docs:
    if d["kind"] == "Job":
        d["metadata"]["name"] = "qwen38fn-ursu-publish-640"
print(yaml.safe_dump_all(docs))
PY
  local j=qwen38fn-ursu-publish-640
  log "publicar $NAME: Job $j"
  espera_job $j && log "publicar $NAME: ok" \
    || { log "publicar $NAME: FALLA"; kubectl logs -n llm job/$j --tail=30; return 1; }
}

tabla_ple() {  # una Job por nodo con su rango, leyendo la vista por el mismo volumen que su rango del residente
  for nr in "gx10-ec3d 0 pvc:hf-cache-nvidia-dgx-ro" "nvidia-dgx 1 host:/home/dibanerz/.cache/huggingface" "dgx3 2 pvc:hf-cache-dgx1-dgx3-llm-ro"; do
    set -- $nr
    python3 - $C/publicar-ursu-tp3.yaml $1 $2 $3 $NAME <<'PY' | tolera | kubectl apply -f - >/dev/null
import sys, yaml
src, node, rank, vol, name = sys.argv[1:]
j = next(d for d in yaml.safe_load_all(open(src)) if d and d["metadata"]["name"] == "qwen38fn-ursu-ple-table-0")
j["metadata"]["name"] = "qwen38fn-ursu-ple-table-" + rank
s = j["spec"]["template"]["spec"]; s["nodeSelector"] = {"kubernetes.io/hostname": node}
c = s["containers"][0]
for e in c["env"]:
    if e["name"] == "TP_RANK": e["value"] = rank
    if e["name"] == "CKPT_DIR": e["value"] = "/cache/huggingface/" + name
kind, ref = vol.split(":", 1)
s["volumes"][0] = {"name": "view", **({"persistentVolumeClaim": {"claimName": ref, "readOnly": True}} if kind == "pvc" else {"hostPath": {"path": ref, "type": "Directory"}})}
print(yaml.safe_dump(j))
PY
  done
  for r in 0 1 2; do
    espera_job qwen38fn-ursu-ple-table-$r 3600 && log "tabla PLE rango $r: ok" \
      || { log "tabla PLE rango $r: FALLA"; kubectl logs -n llm job/qwen38fn-ursu-ple-table-$r --tail=20; return 1; }
  done
  for h in dgx2 dgx dgx3; do log "$h: $(ssh $h "python3 -c 'import json; r=json.load(open(\"/home/user/.cache/qwen38fn-ple-ursu/ready.json\")); print(r[\"tp_rank\"], r[\"checkpoint\"][:8], r[\"bytes\"])'")"; done
}

rueda_b12x() {
  ssh dgx 'mkdir -p /home/dibanerz/.cache/huggingface/wheelhouse'
  rsync -a --chmod=a+rX $D/mtp/b12x/wheels/ dgx:/home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/
  [ "$(ssh dgx sha256sum /home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/b12x-1.3.0-py3-none-any.whl | cut -c1-64)" = \
    c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3 ] && log "rueda b12x: ok" || { log "rueda b12x: sha256 distinto"; return 1; }
}

e768() {
  # el servidor primero: si el Job arranca antes de que haya endpoint, la descarga da timeout y el Job falla (pasó el 13-10)
  tolera < $C/publicar-ursu-e768.yaml | python3 -c 'import sys,yaml; print(yaml.safe_dump_all([d for d in yaml.safe_load_all(sys.stdin) if d and d["kind"] != "Job"]))' | kubectl apply -f - >/dev/null
  kubectl rollout status -n llm deploy/qwen38fn-ursu-e768-serve --timeout=600s >/dev/null
  tolera < $C/publicar-ursu-e768.yaml | kubectl apply -f - >/dev/null; log "publicar e768: Job qwen38fn-ursu-e768-publish"
  espera_job qwen38fn-ursu-e768-publish && log "publicar e768: ok" \
    || { log "publicar e768: FALLA"; kubectl logs -n llm job/qwen38fn-ursu-e768-publish --tail=30; return 1; }
}

[ "${1:-}" = e768 ] && { e768 && log "previo: fin" || log "previo: FALLA"; exit; }
rueda_b12x && publica && tabla_ple && e768 && log "previo: fin" || log "previo: FALLA"
