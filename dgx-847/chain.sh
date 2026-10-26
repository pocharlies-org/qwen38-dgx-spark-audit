#!/usr/bin/env bash
# DGX-847: batería contra una receta de ursu, en serie: parser de tools (+ Claude Code si hay /v1/messages) -> decode v4 ->
# benchy medium -> calidad (cortada en DEADLINE). /health antes de cada paso; si cae, para.
#   chain.sh 2x|1x   (DEADLINE=epoch en el entorno)
set -u
ARM=$1; D=/home/user/bench-results/dgx-847/$ARM; mkdir -p $D
case $ARM in
  2x) RES=http://spark2:8000; DEPLOY=ursu-2x-head ;;
  1x) RES=http://spark3:8000; DEPLOY=ursu-1x ;;
esac
export RES DEPLOY ARM BASE=$D OUT=$D/decode MODEL=qwen3.8-flash-next DEADLINE KUBECONFIG=/home/user/.kube/config
ts(){ date -u +%H:%M:%SZ; }
health(){ c=$(curl -s -o /dev/null -w '%{http_code}' -m 10 $RES/health); echo "$(ts) health antes de $1: $c"; [ "$c" = 200 ] || { kubectl logs -n llm deploy/$DEPLOY --tail=60; echo "$(ts) FIN: caído antes de $1"; exit 2; }; }

health tools
python3 /home/user/bench-results/dgx-847/tools-check.py $RES $D; echo "$(ts) tools exit=$?"
if grep -q '"anthropic_messages": 200' $D/tools-check.json; then
  W=$(mktemp -d /tmp/dgx847-cc-XXXX); echo "hola desde DGX-847" > $W/entrada.txt
  ( cd $W && ANTHROPIC_BASE_URL=$RES ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN= CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 600 \
    claude -p --model $MODEL --allowedTools "Read Write Bash(ls:*)" --permission-mode acceptEdits --output-format json \
    "Lee entrada.txt, escribe su contenido en mayúsculas en salida.txt y luego lista el directorio con ls." ) > $D/claude-code.json 2>&1
  echo "$(ts) claude-code exit=$? salida.txt=$(cat $W/salida.txt 2>/dev/null || echo NO)"
fi

health decode
python3 /home/user/bench-results/dgx-847/measure.py > $D/run-decode.log 2>&1; echo "$(ts) decode exit=$? ($(head -2 $D/decode/DONE-base 2>/dev/null | tr '\n' ' '))"
health benchy
python3 /home/user/bench-results/dgx-847/launch-benchy-quality.py > $D/run-benchy-quality.log 2>&1; echo "$(ts) benchy+calidad exit=$? ($(head -2 $D/DONE-benchy-quality | tr '\n' ' '))"
echo "$(ts) FIN"
