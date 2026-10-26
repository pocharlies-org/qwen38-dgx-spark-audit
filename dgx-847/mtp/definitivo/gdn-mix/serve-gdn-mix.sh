#!/bin/bash
# DGX-852 · QWEN_GDN_MIX: entrypoint que parchea la imagen de ursu ANTES de
# `vllm serve` y luego lanza su receta tal cual (serve-2x.sh / serve-1x.sh),
# igual que hacemos en launch-ursu-b12x.sh con los parches de TP=3.
#
# Montar este fichero y patch_gdn_mix.py en el ConfigMap ursu-recipe (junto a
# serve-2x.sh / serve-1x.sh) y cambiar en ursu.yaml:
#     args: [bash, /etc/ursu/serve-gdn-mix.sh]
# anadiendo en env:
#     - {name: QWEN_GDN_MIX, value: "A"}     # o "B"
#     - {name: GDN_TP,       value: "2"}       # 2 -> serve-2x.sh, 1 -> serve-1x.sh
#
# El parcheador es FAIL-CLOSED: si QWEN_GDN_MIX=A|B y un ancla no casa, sale
# != 0 y aqui abortamos sin lanzar vllm (el pod no arranca, en vez de servir una
# configuracion a medias). Con QWEN_GDN_MIX vacio no parchea y lanza la receta
# original (comportamiento de siempre), util como escape de rollback.
set -euo pipefail

MIX="${QWEN_GDN_MIX:-}"
case "$MIX" in
  ""|A|B|B2) ;;
  *) echo "QWEN_GDN_MIX ABORTA: valor invalido '$MIX' (esperado A, B, B2 o vacio)" >&2; exit 2 ;;
esac

PATCH="${GDN_MIX_PATCH:-/etc/ursu/patch_gdn_mix.py}"
if [ -n "$MIX" ]; then
  [ -f "$PATCH" ] || { echo "QWEN_GDN_MIX ABORTA: falta el parcheador $PATCH" >&2; exit 2; }
  # -I: no carga codigo desde el cwd (los ConfigMaps se montan con symlinks).
  # El parcheador localiza la imagen con find_spec, sin importar vllm (no toca CUDA).
  python3 -I "$PATCH" || { echo "QWEN_GDN_MIX ABORTA: el parcheador fallo (ancla o sintaxis)" >&2; exit 3; }
  echo "QWEN_GDN_MIX=$MIX: imagen parcheada"
else
  # Vacio = paso puro: ni parche ni flags. Con este entrypoint montado de forma
  # permanente, QWEN_GDN_MIX="" reproduce EXACTAMENTE el arranque de hoy.
  echo "QWEN_GDN_MIX vacio: sin parchear y sin flags, lanzo la receta original"
  exec bash "/etc/ursu/serve-${GDN_TP:-2}x.sh"
fi

# Args comunes a A y B: estado BF16 + decode b12x + prefill triton. La receta ya
# trae --gdn-decode-kernel b12x y --mamba-cache-mode align; aqui fijamos el
# prefill a triton y el dtype del estado, y repetimos el decode por claridad.
# B2 NO fuerza prefill triton: con --gdn-decode-kernel b12x la politica de vLLM
# (qwen_gdn_linear_attn.py:328-335) ya resuelve prefill=b12x, que es lo que crea
# la ruta mixta `_forward_core_b12x_mixed`. B2 es la receta de ursu (b12x/b12x)
# con el estado en BF16; el parcheador hace que el prefill b12x acepte BF16.
if [ "$MIX" = "B2" ]; then
  GDN_ARGS='--mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel b12x --additional-config {"gdn_prefill_backend":"b12x","gdn_decode_kernel":"b12x"}'
  # el additional-config pisa el de la receta 3x (triton/triton); en 2x y 1x no hay otro y b12x es lo que ya resolvía
  # Sus semillas de ajuste son de FP32 (el hash de la consulta lleva el dtype): con autoajuste, el primer arranque
  # ajusta ~700 variantes GDN en BF16, 4-7 h por nodo con la memoria al limite (14-10 12:25Z). Apagado, b12x usa la
  # config por defecto SOLO en lo que no esta en cache (el GDN BF16); B2_AUTOTUNE=1 hace el ajuste completo.
  export B12X_AUTOTUNE=${B2_AUTOTUNE:-0}
else
  GDN_ARGS='--mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel b12x --additional-config {"gdn_prefill_backend":"triton"}'
fi

# B y B2 = registros compactos + checkpoints diferidos (el parcheador habilito
# las relaxaciones en modo B/B2). A los deja explicitamente apagados.
if [ "$MIX" = "B" ] || [ "$MIX" = "B2" ]; then
  export VLLM_GDN_DEFERRED_CHECKPOINTS=1
  export VLLM_GDN_COMPACT_RECORDS=1
else
  export VLLM_GDN_DEFERRED_CHECKPOINTS=0
  export VLLM_GDN_COMPACT_RECORDS=0
fi

# EXTRA_ARGS va al FINAL de `vllm serve` (la receta lo anexa despues de sus
# flags) y argparse se queda con el ultimo valor, asi que ANEXAMOS los nuestros
# al EXTRA_ARGS que ya hubiera: los nuestros ganan sobre la receta y sobre un
# EXTRA_ARGS heredado (p. ej. el de E2, si alguien reutiliza su transform).
# Ojo: `--additional-config` no se fusiona, se sustituye -- no pases otro
# additional-config en EXTRA_ARGS cuando lances A o B.
export EXTRA_ARGS="${EXTRA_ARGS:-} ${GDN_ARGS}"

PREFILL=$([ "$MIX" = "B2" ] && echo b12x || echo triton)
echo "QWEN_GDN_MIX=$MIX decode=b12x prefill=$PREFILL ssm=bf16 deferred=$VLLM_GDN_DEFERRED_CHECKPOINTS compact=$VLLM_GDN_COMPACT_RECORDS autotune=${B12X_AUTOTUNE:-1} tp=${GDN_TP:-2}"
exec bash "/etc/ursu/serve-${GDN_TP:-2}x.sh"
