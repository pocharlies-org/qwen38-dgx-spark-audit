#!/usr/bin/env bash
# DGX-852 · Lanza A0 / A / B sobre la imagen de ursuciprian y captura el greedy.
# LO EJECUTA LA SESIÓN DUEÑA DE LOS SPARKS (toca el clúster); esta sesión no.
#
#   prueba-gdn-mix.sh A            # A en el 2x (dgx2+dgx1), R2
#   prueba-gdn-mix.sh B 1          # B en el 1x (dgx3), R1
#   prueba-gdn-mix.sh B2 2         # B2 (prefill+decode b12x, estado BF16) en el 2x
#   prueba-gdn-mix.sh A 2 e2-1.json  # A + compare contra la captura de E2
#
# B2 es la B servible: prefill y decode b12x con estado BF16 (el parcheador hace
# que el kernel de prefill acepte BF16). NO aborta a c2+ como B1, porque los
# lotes mixtos van por la ruta b12x_mixed disenada. El paso 8 (aborto) es solo B.
#
# A0 es el control sin parche: decode CUDA fusionado + prefill triton, que la
# política de vLLM ya permite (qwen_gdn_linear_attn.py:328-335 solo mira
# "b12x"). Si el op no está compilado en su imagen, vLLM NO aborta: loguea
# "Falling back to the Triton GDN decode path" (línea 818) y sirve E2 disfrazado
# -- por eso A0 se juzga por el log, no por el número.
#
# Antes de nada, el residente a 0 y la memoria devuelta, igual que prueba-e2.sh.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
G=$D/mtp/definitivo/gdn-mix
MODE=${1:?modo: A0|A|B|B2}
TP=${2:-2}
BASELINE=${3:-}
case "$MODE" in E2|A0|A|B|B2) ;; *) echo "modo $MODE: esperado E2, A0, A, B o B2" >&2; exit 2 ;; esac
N=${N:-$MODE$([ "$TP" = 1 ] && echo -tp1)}  # nombre de las capturas: las de TP=1 no pisan las de TP=2
mkdir -p $G/capturas
R=$([ "$TP" = 1 ] && echo http://spark3:8000 || echo http://spark2:8000)
SEL=$([ "$TP" = 1 ] && echo 'app=ursu-1x' || echo 'app in (ursu-2x-head,ursu-2x-worker)')

# 1. Clúster libre: residente a 0 y memoria unificada devuelta (prueba-e2.sh:13-20).
kubectl scale deploy -n llm $DEPS --replicas=0 >/dev/null
while kubectl get pods -n llm --no-headers | grep -q '^qwen38-flash-next-'; do sleep 5; done
# 14-10 08:50Z: solo se baja lo que se rehace (la otra receta puede seguir arriba) y juntos, por la cita del TCPStore
if [ "$TP" = 1 ]; then HS=dgx3; kubectl scale -n llm deploy/ursu-1x --replicas=0 >/dev/null
else HS="dgx dgx2"; kubectl scale -n llm deploy/ursu-2x-head deploy/ursu-2x-worker --replicas=0 >/dev/null; fi
while kubectl get pods -n llm -l "$SEL" --no-headers 2>/dev/null | grep -q .; do sleep 5; done
for h in $HS; do
  t0=$(date +%s)
  until [ "$(ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")" -ge 100 ]; do
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "gdn-mix: $h no recupera memoria en 15 min"; exit 1; }; sleep 10
  done
done

# 2. Receta + parcheador dentro del ConfigMap ursu-recipe (se montan en /etc/ursu).
kubectl create configmap -n llm ursu-recipe \
  --from-file=$O/recipe/serve-2x.sh --from-file=$O/recipe/serve-1x.sh \
  --from-file=$G/serve-gdn-mix.sh --from-file=$G/patch_gdn_mix.py \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap -n llm ursu-env-2x --from-env-file=$O/recipe/env-2x.env \
  --from-env-file=$O/recipe/env-sparkrun-2x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap -n llm ursu-env-1x --from-env-file=$O/recipe/env-1x.env \
  --from-env-file=$O/recipe/env-sparkrun-1x.env --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if [ "${REFUSAL:-0}" = 1 ]; then  # detector de negativas + dial de abliteration de producción (refusal-port/PLAN.md)
  RP=$D/refusal-port
  kubectl create configmap -n llm qwen38fn-b12x-root --from-file=$RP/install.py --from-file=$RP/run.sh \
    --from-file=$RP/base-hashes.json --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl create configmap -n llm qwen38fn-b12x-refusal $(printf -- '--from-file=%s ' $RP/refusal/*.py $RP/refusal/*.npy) \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl create configmap -n llm qwen38fn-b12x-refusal-payload $(printf -- '--from-file=%s ' $RP/refusal/payload/*.py) \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
fi

# 3. Ursu.yaml: solo el TP pedido. A|B cambian el entrypoint y ponen QWEN_GDN_MIX
#    (los flags y el deferred/compact los arma serve-gdn-mix.sh). A0 no parchea:
#    lleva los flags por EXTRA_ARGS, como prueba-e2.sh.
MODE=$MODE TP=$TP python3 - $O/ursu.yaml <<'PY' | kubectl apply -f - >/dev/null
import os, sys, yaml
mode, tp = os.environ["MODE"], os.environ["TP"]
want = {"ursu-1x"} if tp == "1" else {"ursu-2x-head", "ursu-2x-worker"}
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
out = []
for d in docs:
    name = d["metadata"]["name"]
    if name not in want:
        continue
    c = d["spec"]["template"]["spec"]["containers"][0]
    c.setdefault("env", []).append(
        {"name": "VLLM_MTP_DRAFT_VOCAB", "value": "/opt/mtp-vocab-es/ids-es-K131072.txt.gz"})
    if mode in ("A0", "E2"):  # E2 = la base (prueba-e2.sh): BF16 + GDN triton/triton
        c["env"] += [
            {"name": "EXTRA_ARGS", "value": '--mamba-ssm-cache-dtype bfloat16 ' + {
             "A0": '--gdn-decode-kernel cuda --additional-config {"gdn_prefill_backend":"triton"}',
             "E2": '--gdn-decode-kernel triton --additional-config {"gdn_prefill_backend":"triton","gdn_decode_kernel":"triton"}'}[mode]},
            {"name": "VLLM_GDN_DEFERRED_CHECKPOINTS", "value": "0"},
            {"name": "VLLM_GDN_COMPACT_RECORDS", "value": "0"}]
        c["volumeMounts"].append({"name": "mtp-vocab-es", "mountPath": "/opt/mtp-vocab-es", "readOnly": True})
        d["spec"]["template"]["spec"]["volumes"].append(
            {"name": "mtp-vocab-es", "configMap": {"name": "qwen38fn-mtp-draft-vocab"}})
    else:
        c["args"] = ["bash", "/etc/ursu/serve-gdn-mix.sh"]
        c["env"] += [{"name": "QWEN_GDN_MIX", "value": mode}, {"name": "GDN_TP", "value": tp}]
        if os.environ.get("EXTRA"):  # serve-gdn-mix.sh lo antepone a sus flags y la receta lo anexa al final: gana
            c["env"].append({"name": "EXTRA_ARGS", "value": os.environ["EXTRA"]})
        c["volumeMounts"].append({"name": "mtp-vocab-es", "mountPath": "/opt/mtp-vocab-es", "readOnly": True})
        d["spec"]["template"]["spec"]["volumes"].append(
            {"name": "mtp-vocab-es", "configMap": {"name": "qwen38fn-mtp-draft-vocab"}})
        if os.environ.get("REFUSAL") == "1":  # run.sh instala el mod (fail-closed) y cede a serve-gdn-mix.sh
            c["args"] = ["bash", "/opt/ursu-mod/run.sh", "/etc/ursu/serve-gdn-mix.sh"]
            c["env"] += [{"name": k, "value": v} for k, v in (
                ("VLLM_REFUSAL_DIRECTION", "/opt/refusal/refusal_direction_flashnext.npy"), ("VLLM_REFUSAL_LAMBDA", "0"),
                ("VLLM_REFUSAL_PROBE", "1"), ("VLLM_REFUSAL_PROBE_LAYERS", "16,20,24,28,32"),
                ("VLLM_REFUSAL_PROBE_CAL", "/opt/refusal-probe-cal/cal.json"))]
            for n, m, cm in (("b12x-mod", "/opt/ursu-mod", "qwen38fn-b12x-root"),
                             ("b12x-refusal", "/opt/ursu-mod/refusal", "qwen38fn-b12x-refusal"),
                             ("b12x-refusal-payload", "/opt/ursu-mod/refusal/payload", "qwen38fn-b12x-refusal-payload"),
                             ("probe-cal", "/opt/refusal-probe-cal", "qwen38fn-refusal-probe-cal")):
                c["volumeMounts"].append({"name": n, "mountPath": m, "readOnly": True})
                d["spec"]["template"]["spec"]["volumes"].append({"name": n, "configMap": {"name": cm}})
    out.append(d)
print(yaml.safe_dump_all(out))
PY
log "gdn-mix: $MODE en TP=$TP ($R) arrancando"

# 4. Salud: el primer reinicio ya es fallo (una config mala se repite).
t0=$(date +%s)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 $R/health)" = 200 ]; do
  kubectl get pods -n llm -l "$SEL" -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' \
    | grep -qE '[1-9]' && { log "gdn-mix: $MODE se reinicio"; kubectl logs -n llm -l "$SEL" --tail=60; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "gdn-mix: $MODE sin health en 60 min"; kubectl logs -n llm -l "$SEL" --tail=60; exit 1; }
  sleep 15
done
log "gdn-mix: $MODE arriba en $(( ($(date +%s) - t0) / 60 )) min"

# 5. Confirmacion de que el mix es el que se pidio (no un E2 disfrazado).
L=$G/capturas/$N-arranque.log
kubectl logs -n llm -l "$SEL" --tail=-1 > $L 2>&1  # con -l el tope por defecto son 10 líneas por pod
log "gdn-mix: selector -> $(grep -h 'GDN decode kernel' $L | sed 's/^.*\] //' | sort -u | paste -sd' ')"
grep -q 'Falling back to the Triton GDN decode path' $L \
  && log "gdn-mix: OJO, el kernel pedido no se pudo usar y vLLM cayo a triton"
grep -q 'QWEN_GDN_MIX' $L && log "gdn-mix: parche -> $(grep -h 'QWEN_GDN_MIX' $L | sed 's/^.*\] //' | head -2 | paste -sd' | ')"
grep -qE 'GDN (deferred checkpoints|compact records) enabled' $L \
  && log "gdn-mix: diferidos -> $(grep -hE 'GDN (deferred|compact)' $L | sed 's/^.*\] //' | tr '\n' ' ' | cut -c1-160)"

# 6. Dos capturas greedy seguidas: la segunda contra la primera es el SUELO DE
#    RUIDO del batching; sin ella no se puede juzgar ninguna divergencia.
for n in 1 2; do
  python3 -I $G/greedy_compare.py --url $R --model qwen3.8-flash-next \
    --long-tokens 16000 --conc 8 --capture $G/capturas/$N-$n.json \
    > $G/capturas/$N-$n.log 2>&1 || { log "gdn-mix: capture $n fallo"; tail -5 $G/capturas/$N-$n.log; exit 1; }
  log "gdn-mix: captura $n -> $G/capturas/$N-$n.json"
done

# 7. Compare contra la baseline de E2 (si se paso).
if [ -n "$BASELINE" ]; then
  python3 -I $G/greedy_compare.py --url $R --model qwen3.8-flash-next \
    --compare $BASELINE --noise $G/capturas/$(basename $BASELINE .json | sed 's/-1$//')-2.json \
    --out $G/capturas/$N-vs-e2.json > $G/capturas/$N-vs-e2.log 2>&1
  log "gdn-mix: compare exit=$? $(tail -2 $G/capturas/$N-vs-e2.log | head -1)"
fi

# 8. B: el lote mixto a c8 DEBE abortar (guard compact+spec, qwen_gdn_linear_attn.py:3010-3014).
#    Se hace al final porque tira el pod: es la prueba, no un fallo.
if [ "$MODE" = B ]; then
  log "gdn-mix: B a c2 con prefill en vuelo (se espera RuntimeError de registros compactos)"
  python3 -I $G/greedy_compare.py --url $R --model qwen3.8-flash-next \
    --long-tokens 16000 --conc 2 --capture $G/capturas/B-mixto.json > $G/capturas/B-mixto.log 2>&1
  log "gdn-mix: B-mixto exit=$? pod=$(kubectl get pods -n llm -l "$SEL" -o jsonpath='{.items[*].status.phase}')"
  kubectl logs -n llm -l "$SEL" 2>&1 | grep -m2 'compact GDN records' | sed 's/^.*\] //'
fi
log "gdn-mix: $MODE terminado; capturas en $G/capturas"
