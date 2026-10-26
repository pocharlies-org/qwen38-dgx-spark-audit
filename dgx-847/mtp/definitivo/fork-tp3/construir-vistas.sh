#!/usr/bin/env bash
# DGX-852: las DOS vistas que necesita el fork de ursuciprian para arrancar en TP=3, construidas
# EN dgx1 y en ese orden. Nada de esto toca Kubernetes ni las vistas existentes (solo las lee y
# las enlaza). Cero GPU: es CPU + I/O, y va a prioridad idle para no perturbar lo que corra.
#
#  1. qwen38fn-mtp-ursu-nvfp4-tp3-fork  <- --fork-view sobre la e768: vocabulario recortado a
#     248.320 filas, PLE re-troceada a 128 partes de 2.500.014 filas (256 filas a 0 al FINAL de la
#     tabla global) y config con el vocabulario real. 17 ficheros nuevos (52 GiB), 19 enlazados.
#  2. qwen38fn-mtp-ursu-mxfp8-tp3       <- --mxfp8-copy sobre el shard 35 de local-inference-lab
#     (la copia que sirve lotes >= 41 tokens): el mismo relleno TP=3 que la vista principal, porque
#     `_tp_shard` corta la copia contra la partición de la capa NVFP4 y compara las formas.
#
# Root: la fuente e768 es root:root y `fs.protected_hardlinks=1` no deja enlazar un fichero que no
# posees (EPERM). `place()` aborta si el enlace falla en lugar de copiar 77 GB a ciegas, así que
# uid 1000 no puede construir esta vista. Se mantienen el techo de memoria, nice 19 e ionice idle.
set -euo pipefail
H=/home/dibanerz/dgx-852
CACHE=/home/dibanerz/.cache/huggingface
E768=$CACHE/qwen38fn-mtp-ursu-nvfp4-tp3-e768
FORK=$CACHE/qwen38fn-mtp-ursu-nvfp4-tp3-fork
HUB=$CACHE/hub
COPY=$(echo $HUB/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/.)
MXFP8=$CACHE/qwen38fn-mtp-ursu-mxfp8-tp3
ORIGCFG=$(echo $HUB/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/*/config.json)
MARK=.built-dgx852-fork-tp3

guard() {  # el proceso pesado, con el techo de memoria y la prioridad que pidió Dani
  sudo -n systemd-run --scope -q -p MemoryMax=12G --setenv=HOME=/home/dibanerz \
    nice -n 19 ionice -c3 python3 "$H/correr.py" "$@"
}

[ -d "$E768" ] || { echo "no existe la fuente $E768"; exit 1; }
if [ "${1:-}" != "--fuerza" ]; then  # re-construir 56 GB por accidente mientras corre otra prueba
  for d in "$FORK" "$MXFP8"; do
    if [ -e "$d" ]; then echo "ya existe $d — para reconstruir: $0 --fuerza"; exit 1; fi
  done
fi
[ -f "$ORIGCFG" ] || { echo "no hay config original en $HUB"; exit 1; }
libre=$(df -B1 --output=avail "$CACHE" | tail -1)
total=$(df -B1 --output=size "$CACHE" | tail -1)
# El kubelet de los Sparks eviction-hard por defecto a 10% de nodefs y 15% de imagefs: escribir aqui por
# encima de ese limite mete al nodo en DiskPressure, desaloja lo que este corriendo y GC de las imagenes.
# La vista escribe 56 GB: hace falta que DESPUES del escrito queden >20% del volumen libre.
[ "$libre" -gt 65000000000 ] || { echo "NO CABE: $libre bytes libres; la vista fork escribe 56 GB"; exit 1; }
umbral=$(( total / 5 ))
[ $((libre - 60000000000)) -gt "$umbral" ] || {
  echo "NO: tras escribir 56 GB quedarian $(( (libre - 60000000000) / 1000000000 )) GB de $((total / 1000000000))" \
       "(hacen falta > $((umbral / 1000000000)) GB para no meter el nodo en DiskPressure)." \
       "Libera espacio primero (vistas viejas, hub) o construye en otro nodo."; exit 1; }
echo "== dgx1 $(date -Is) | libres $((libre / 1000000000)) GB | fuente $E768 | copia $COPY"

echo "== 1/2 plan (check, sin escribir)"
guard check --src "$E768" --fork-view 2>&1 | grep -v "^AVISO (config distinta"

echo "== 2/2 convert --fork-view"
guard convert --src "$E768" --out "$FORK" --fork-view --mark "$MARK"
echo "== verify --fork-view"
guard verify --src "$E768" --out "$FORK" --fork-view

echo "== convert --mxfp8-copy (shard 35 de la copia MXFP8)"
guard convert --src "$COPY" --out "$MXFP8" --mxfp8-copy --config-from "$ORIGCFG" --mark "$MARK"
echo "== verify --mxfp8-copy"
guard verify --src "$COPY" --out "$MXFP8" --mxfp8-copy --config-from "$ORIGCFG"

echo "== pruebas"
python3 - "$FORK" "$E768" "$MXFP8" "$COPY" <<'PY'
import json, os, struct, sys
fork, e768, mxfp8, copy = sys.argv[1:5]
tc = json.load(open(f"{fork}/config.json"))["text_config"]
print("  config fork: vocab_size", tc["vocab_size"], "| tp3_orig_vocab_size fuera:",
      "tp3_orig_vocab_size" not in tc, "| divisor", tc["make_ngram_vocab_size_divisible_by"],
      "| dtype PLE", tc["ple_embedding_dtype"], "| moe", tc["moe_intermediate_size"])
for d, s in ((fork, e768), (mxfp8, copy)):
    n = sum(1 for f in os.listdir(d) if f.endswith(".safetensors"))
    enl = sum(1 for f in os.listdir(d) if f.endswith(".safetensors")
              and os.path.isfile(f"{s}/{f}") and os.stat(f"{d}/{f}").st_ino == os.stat(f"{s}/{f}").st_ino)
    print(f"  {os.path.basename(d)}: {n} shards, {enl} ENLAZADOS DUROS a su fuente")
def head(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))
h = head(f"{fork}/model-00002-of-00036.safetensors")
k = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
print("  PLE en la vista:", h[k]["dtype"], h[k]["shape"], "| en la fuente:",
      head(f"{e768}/model-00002-of-00036.safetensors")[k]["shape"])
h34 = head(f"{fork}/model-00034-of-00036.safetensors")
print("  embed/lm_head:", h34["model.language_model.embed_tokens.weight"]["shape"],
      h34["lm_head.weight"]["shape"], "| escala global PLE:",
      h34["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"])
h35 = head(f"{mxfp8}/model-00035-of-00036.safetensors")
print("  copia MXFP8:", {k: (h35[k]["dtype"], h35[k]["shape"]) for k in
      ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
       "model.language_model.layers.0.linear_attn.out_proj.weight",
       "model.language_model.layers.0.linear_attn.in_proj_a.weight")})
print("  index copia:", len(json.load(open(f"{mxfp8}/model.safetensors.index.json"))["weight_map"]),
      "tensores; config.json presente:", os.path.exists(f"{mxfp8}/config.json"))
PY
df -h "$CACHE" | tail -1
echo "== hecho $(date -Is): MODEL_DIR=$FORK  VLLM_B12X_NVFP4_MXFP8_CHECKPOINT=$MXFP8"
