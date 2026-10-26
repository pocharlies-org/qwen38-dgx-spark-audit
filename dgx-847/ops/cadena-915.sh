#!/usr/bin/env bash
# DGX-915: k8s-ai#183 -> merge -> repin de ai en gitops -> merge -> sync -> arranque medido del TP=2. No toca pods.
set -u
export KUBECONFIG=~/.kube/config
OPS=/home/user/bench-results/dgx-847/ops; OLD=d8399156fb51152214e074f6fe130a0dcc654f03
ts() { date -u +%T; }
$OPS/espera-lista.sh pocharlies-org/k8s-ai-pocharlies 183 3600; rc=$?; [ $rc -eq 0 ] || { echo "FALLO puerta k8s-ai rc=$rc"; exit $rc; }
SHA=$(gh pr view 183 -R pocharlies-org/k8s-ai-pocharlies --json mergeCommit -q .mergeCommit.oid); echo "$(ts) k8s-ai merge $SHA"
G=~/k8s/k8s-gitops-pocharlies; W=$G/.claude/worktrees/DGX-915-repin-ai
git -C $G fetch -q origin && git -C $G worktree add -q -b DGX-915-repin-ai $W origin/deploy/prod || { echo "FALLO worktree gitops"; exit 1; }
python3 - "$W/apps/ai.yaml" "$OLD" "$SHA" <<'PY' || { echo "FALLO edit ai.yaml"; exit 1; }
import sys
p, old, new = sys.argv[1:]; s = open(p).read(); line = f"    targetRevision: {old}\n"
assert s.count(line) == 1, "pin no encontrado"
s = s.replace(line, f"""    # 15-10-2026 (DGX-915): {new[:7]} (k8s-ai #183, misma línea y path): KV fija del TP=2 a 3,7 M tokens (30,535e9 bytes por
    # rango; con 4,2 M el head arrancaba en ~12 min por falta de memoria libre en dgx2). Reinicia head y worker; el TP=1 de dgx3
    # no se toca. La voz baja y sube con los hooks.
    # ROLLBACK de esto: {old} con el mismo path.
    targetRevision: {new}
""")
open(p, "w").write(s)
PY
git -C $W commit -qam "DGX-915: ai a ${SHA:0:7}, KV fija del TP=2 a 3,7 M tokens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git -C $W push -q -u origin DGX-915-repin-ai 2>&1 | grep -vE "value-absent|if it was renamed|^remote:" | tail -3
N=$(gh pr create -R pocharlies-org/k8s-gitops-pocharlies --base deploy/prod --head DGX-915-repin-ai \
  --title "DGX-915 · ai a ${SHA:0:7}: KV fija del servidor grande a 3,7 M tokens" --body "## Qué cambia

\`ai\` pasa de \`${OLD:0:7}\` a \`${SHA:0:7}\`, en la misma línea \`pin/qwen38-ursu-2mas1-20261014\` y el mismo path. Trae k8s-ai #183:

- **Servidor grande (TP=2):** caché KV fija de 3,7 M tokens (30,535e9 bytes por rango). Antes eran 4,2 M (34,66e9), que arrancaban en ~12 min porque dgx2 se quedaba sin memoria libre.
- **Servidor pequeño (TP=1, dgx3):** sin cambios y sin reinicio.

## Efecto al desplegar

- Se reinician head y worker del grande. Mientras arranca (se espera ~5 min), el fallback de LiteLLM manda \`qwen38-flash-next\` a Alibaba y \`tooling\` da 503.
- La voz baja y sube con los hooks de la sync.

## Rollback

\`targetRevision: $OLD\` con el mismo path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)" | grep -oE '[0-9]+$')
echo "$(ts) gitops PR #$N"; [ -n "$N" ] || exit 1
$OPS/espera-lista.sh pocharlies-org/k8s-gitops-pocharlies $N 3600; rc=$?; [ $rc -eq 0 ] || { echo "FALLO puerta gitops rc=$rc"; exit $rc; }
echo "$(ts) gitops merge"; git -C $G worktree remove $W && git -C $G branch -D DGX-915-repin-ai >/dev/null
# Arranque: sin tocar nada. stt-turbo se registra por DGX-913.
fin=$((SECONDS + 3600)); stt_prev=""; head_pod=""
while (( SECONDS < fin )); do
  stt=$(kubectl get deploy -n llm stt-turbo -o jsonpath='{.spec.replicas}'); [ "$stt" != "$stt_prev" ] && { echo "$(ts) stt-turbo replicas=$stt"; stt_prev=$stt; }
  p=$(kubectl get pod -n llm -l app=qwen38-flash-next-head -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.annotations.config\.k8s\.e-dani\.com/launch-revision} {.status.startTime} {.status.containerStatuses[0].ready}{"\n"}{end}' | grep dgx915 | head -1)
  if [ -n "$p" ]; then
    set -- $p; [ "$head_pod" != "$1" ] && { head_pod=$1; echo "$(ts) head nuevo $1 start=$3"; }
    if [ "$4" = "true" ]; then
      rdy=$(kubectl get pod -n llm $1 -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}')
      kv=$(kubectl logs -n llm $1 2>/dev/null | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1)
      echo "$(ts) head READY start=$3 ready=$rdy · $kv"; break
    fi
  fi
  sleep 10
done
[ -n "${rdy:-}" ] || { echo "$(ts) TOPE: head sin Ready en 60 min (no se ha tocado nada)"; exit 4; }
fin=$((SECONDS + 1800))
while (( SECONDS < fin )); do
  ph=$(kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["phase"])')
  op=$(kubectl get application -n argocd ai -o jsonpath='{.status.operationState.phase}')
  [ "$ph" = ready ] && [ "$op" = Succeeded ] && break; sleep 15
done
echo "$(ts) arbitro=$ph op=$op"; kubectl get deploy -n llm stt-turbo omnivoice-tts-nvfp4 qwen38-flash-next-worker -o custom-columns=N:.metadata.name,R:.spec.replicas,A:.status.availableReplicas --no-headers
echo "dgx2: $(ssh -o ConnectTimeout=5 dgx2 'free -g | sed -n 2p')"; echo "dgx1: $(ssh -o ConnectTimeout=5 dgx 'free -g | sed -n 2p')"
[ "$ph" = ready ] && [ "$op" = Succeeded ]
