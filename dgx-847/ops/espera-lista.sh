#!/usr/bin/env bash
# Uso: espera-lista.sh <org/repo> <pr> [tope_s]. Sondea la puerta; con LISTA fusiona (squash) y sale 0 con el SHA.
# Sale 1 si la puerta habla de checks fallidos, 3 si pide la aprobación de un rol 5 min seguidos, 4 al tope.
set -u
repo=$1; pr=$2; tope=${3:-2400}; A=~/.local/libexec/x86-host-runtime/company-aprobar
fin=$((SECONDS + tope))
while (( SECONDS < fin )); do
  out=$(timeout 120 "$A" estado "$repo#$pr" 2>&1)
  if grep -q 'LISTA para fusionar' <<<"$out"; then
    gh pr merge "$pr" -R "$repo" --squash 2>&1 | tail -3
    # Con `mergeable` aún UNKNOWN el merge puede no entrar y gh no siempre lo dice: sale 0 solo con MERGED (16-10, #1218).
    for _ in 1 2 3 4 5 6; do
      sleep 10
      st=$(gh pr view "$pr" -R "$repo" --json state,mergeCommit -q '.state+" "+(.mergeCommit.oid // "")')
      [ "${st%% *}" = MERGED ] && { echo "FUSIONADA $st"; exit 0; }
    done
    echo "no fusionó ($st); vuelvo a mirar la puerta"
    continue
  fi
  if grep -qiE 'checks? (en rojo|fallid|failed)|FAILURE' <<<"$out"; then echo "$out"; exit 1; fi
  # Una aprobación de rol no llega sola: si la puerta la pide 5 min seguidos, sale para que alguien la lance.
  if grep -q 'falta la aprobación de' <<<"$out"; then
    falta=${falta:-$SECONDS}; (( SECONDS - falta >= 300 )) && { echo "FALTA ROL:"; echo "$out"; exit 3; }
  else unset falta; fi
  sleep 30
done
echo "TOPE sin LISTA:"; echo "$out"; exit 4
