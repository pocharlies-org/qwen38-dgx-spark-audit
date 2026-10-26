#!/usr/bin/env bash
# DGX-852 (Dani 14-10): prefill de un servidor, igual para todas las variantes. $1 URL base, $2 directorio de salida.
#  frio: prompts de 2K y 16K sin caché de prefijo, a 1 y 4 peticiones a la vez.
#  contexto largo: 2K nuevos encima de 64K ya en caché (como DGX-847).
set -u
U=$1; M=$2; mkdir -p $M
B=/home/user/bin/benchy-venv/bin/llama-benchy
C="--base-url $U/v1 --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next --tokenizer /home/user/benchy-tokenizer/qwen38-fn-hybrid --runs 3 --format md"
$B $C --pp 2048 16384 --tg 32 --depth 0 --concurrency 1 4 --no-cache --save-result $M/frio.md > $M/frio.log 2>&1; echo "frio exit=$?"
$B $C --pp 2048 --tg 32 --depth 65535 --concurrency 1 --enable-prefix-caching --save-result $M/contexto64k.md > $M/contexto64k.log 2>&1; echo "contexto64k exit=$?"
