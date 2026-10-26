#!/usr/bin/env bash
# DGX-932: fusiona k8s-ai-pocharlies#185 (solo comentario, sin pin), quita su worktree y, con producción ya de vuelta
# en A (HECHO de medida-b2-tp2.sh), pasa la historia a Done.
set -u
export KUBECONFIG=~/.kube/config JIRA_ROL=cto
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
WT=/home/user/k8s/k8s-ai-pocharlies/.claude/worktrees/DGX-932-b2-autoajuste
out=$(bash /home/user/bench-results/dgx-847/ops/espera-lista.sh pocharlies-org/k8s-ai-pocharlies 185 2700); rc=$?
log "puerta #185 (rc=$rc): $(echo "$out" | tail -1)"
[ $rc = 0 ] || exit 1
git -C /home/user/k8s/k8s-ai-pocharlies worktree remove "$WT" && git -C /home/user/k8s/k8s-ai-pocharlies branch -D DGX-932-b2-autoajuste >/dev/null && log "worktree quitado"
L=/home/user/bench-results/dgx-932/medida-b2-tp2.log; t0=$(date +%s)
until grep -qE '^[0-9:]+Z (HECHO|VUELTA: el TP=2 no)' $L; do
  [ $(( $(date +%s)-t0 )) -gt 4200 ] && { log "FALLO: la vuelta a A no terminó"; exit 1; }
  sleep 30
done
grep -q '^[0-9:]*Z HECHO' $L || { log "FALLO: la vuelta a A no acabó bien: $(grep VUELTA $L | tail -1)"; exit 1; }
python3 ~/.local/libexec/x86-host-runtime/jira_rest.py transicion DGX-932 Done --rol cto 2>&1 | tail -2
log "HECHO"
