#!/usr/bin/env bash
# DGX-852: UNA sola versión del checkpoint ORIGINAL de ursuciprian, con la PLE en NVFP4, que el
# fork carga en TP=1, TP=2 y TP=3. Se construye EN dgx1 (CPU + I/O, cero GPU, cero clúster) y es
# la que sustituye a las tres vistas por TP que llevábamos.
#
#  qwen38fn-mtp-ursu-universal  <- `--universal` sobre el snapshot 16c9bd54 del hub:
#    - PLE NVFP4 RE-TROCEADA (no descuantizada): 128 partes de 2.500.012 -> 2.500.014 filas, tabla
#      320.001.536 -> 320.001.792 (divisor 128 -> 384), 256 filas de relleno AL FINAL de la tabla
#      global (peso 0x00, escala 0x38 = 1,0 en e4m3: ple_layer.py rechaza escalas <= 0).
#    - geometría y expertos rellenos a múltiplos de 3: KV 2->6, GDN 16:48->24:72, shared y MoE
#      640->768. Todos esos números dividen también entre 2 y entre 1: por eso UNA vista sirve los
#      tres tamaños.
#    - vocabulario INTACTO (248.320 filas reales: de ahí sale el hash de la PLE; el fork lo padrea
#      él solo a lcm(64,tp)) y `ple_embedding_dtype: nvfp4`.
#    Lo que no cambia se ENLACEA DURO contra los blobs del hub (mismo inodo, cero duplicado).
#    Medido con `check` sobre el checkpoint real: 35 ficheros reescritos = 119.445.280.392 B
#    (111,3 GiB) y 1 fichero enlazado. La copia MXFP8 NO se rehace: la del TP=3
#    (qwen38fn-mtp-ursu-mxfp8-tp3) ya guarda las formas GLOBALES padreadas y `_tp_shard` la corta
#    por rango, así que vale para 1, 2 y 3. Solo se comprueba, y si no existe se construye.
#
# Root: los blobs del hub son root:root y con `fs.protected_hardlinks=1` enlazar un fichero que no
# posees da EPERM; `place()` aborta en vez de copiar 111 GB a ciegas. Se mantiene el techo de
# memoria y la prioridad idle del otro script (nice 19, ionice idle, MemoryMax).
#
# Uso (en dgx1):  sudo bash /home/dibanerz/dgx-852/construir-universal.sh [--fuerza]
set -euo pipefail
H=/home/dibanerz/dgx-852
CACHE=/home/dibanerz/.cache/huggingface
REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
ORIG=$(echo $CACHE/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$REV/.)
UNI=$CACHE/qwen38fn-mtp-ursu-universal
MXFP8=$CACHE/qwen38fn-mtp-ursu-mxfp8-tp3
COPY=$(echo $CACHE/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/.)
MARK=.built-dgx852-universal
TECHO=80          # porcentaje de uso que no se rebasa al terminar (Dani: hoy 63%)
MIN_LIBRE_PCT=20  # y además tienen que quedar > 20% libres (eviction del kubelet: 10% nodefs)

guard() {  # el proceso pesado, con el techo de memoria y la prioridad que pidió Dani
  sudo -n systemd-run --scope -q -p MemoryMax=12G --setenv=HOME=/home/dibanerz \
    nice -n 19 ionice -c3 python3 "$H/correr.py" "$@"
}

# --- anclas antes de mover un byte -------------------------------------------------------------------
grep -q -- "--universal" "$H/convert-ursu-nvfp4-tp3.py" || {
  echo "NO: $H/convert-ursu-nvfp4-tp3.py no tiene el modo --universal. Copia el conversor de" \
       "mtp/checkpoint/convert-ursu-nvfp4-tp3.py (y su patches/) antes de construir." >&2; exit 1; }
[ -d "$ORIG" ] || { echo "no existe el checkpoint ORIGINAL en $ORIG"; exit 1; }
[ -f "$ORIG/config.json" ] || { echo "no hay config.json en $ORIG"; exit 1; }
if [ "${1:-}" != "--fuerza" ] && [ -e "$UNI" ]; then
  echo "ya existe $UNI — para reconstruir 111 GB por accidente: $0 --fuerza"; exit 1
fi

echo "== dgx1 $(date -Is) | fuente $ORIG"
echo "== plan (check, sin escribir nada)"
# `check` lee solo cabeceras y da la cifra EXACTA de bytes que va a escribir la vista: el guardado
# de disco se decide con ese número, no con un literal que se queda viejo.
PLAN=$(guard check --src "$ORIG" --universal 2>&1 | grep -v "^AVISO (config distinta")
echo "$PLAN" | tail -3
nuevos=$(echo "$PLAN" | grep -oE '[0-9,]+ reescritos' | head -1 | sed -E 's/[^0-9]//g')
[ -n "$nuevos" ] || { echo "NO: 'check' no devolvió la cifra de bytes reescritos" >&2; exit 1; }

libre=$(df -B1 --output=avail "$CACHE" | tail -1)
total=$(df -B1 --output=size "$CACHE" | tail -1)
# El pico no es la suma final: `write_file` escribe cada shard como `.tmp` y luego lo renombra, así
# que durante el último fichero hay 5,1 GB extra en vuelo. Se le añaden 8 GB de colchón (los .tmp
# huérfanos de un abort, el index de 33 MB y SHA256SUMS).
pico=$(( nuevos + 8000000000 ))
queda=$(( libre - pico ))
uso=$(( (total - queda) * 100 / total ))
echo "== disco: ${nuevos} bytes nuevos (pico ${pico}) | libres ahora $(( libre / 1000000000 )) GB de" \
     "$(( total / 1000000000 )) -> tras escribir $(( queda / 1000000000 )) GB libres, uso ${uso}%"
# El kubelet de los Sparks eviction-hard por defecto a 10% de nodefs y 15% de imagefs: escribir por
# encima mete al nodo en DiskPressure, desaloja lo que esté corriendo y GC de las imágenes. Aquí el
# techo es el que puso Dani (80%) y, por debajo, que quede un 20% del volumen.
[ "$queda" -gt 0 ] || { echo "NO CABE: hacen falta $(( pico / 1000000000 )) GB y hay $(( libre / 1000000000 ))" \
     "— libera espacio (vistas viejas, hub) o construye en otro nodo."; exit 1; }
[ "$uso" -lt "$TECHO" ] || { echo "NO: tras escribir la vista el volumen estaría al ${uso}% (techo ${TECHO}%)." \
     "Libera espacio primero."; exit 1; }
[ "$queda" -gt $(( total * MIN_LIBRE_PCT / 100 )) ] || {
  echo "NO: quedarían $(( queda / 1000000000 )) GB libres y hace falta que queden > $(( total * MIN_LIBRE_PCT / 100 / 1000000000 )) GB" \
       "(${MIN_LIBRE_PCT}% del volumen, para el eviction del kubelet)."; exit 1; }

# --- construir ---------------------------------------------------------------------------------------
mkdir -p "$UNI"
echo "== convert --universal"
guard convert --src "$ORIG" --out "$UNI" --universal --mark "$MARK"
echo "== verify --universal (re-troceo byte a byte contra el ORIGINAL y divisibilidad 1/2/3)"
guard verify --src "$ORIG" --out "$UNI" --universal

# --- la copia MXFP8: se reutiliza, y si no, se construye ---------------------------------------------
if [ -f "$MXFP8/model.safetensors.index.json" ]; then
  echo "== copia MXFP8: reutilizo $MXFP8 (sus formas son las GLOBALES padreadas: vale para 1, 2 y 3)"
else
  echo "== copia MXFP8: no existe, la construyo desde $COPY"
  guard convert --src "$COPY" --out "$MXFP8" --mxfp8-copy --config-from "$ORIG/config.json" --mark "$MARK"
  guard verify --src "$COPY" --out "$MXFP8" --mxfp8-copy --config-from "$ORIG/config.json"
fi

echo "== pruebas"
python3 - "$UNI" "$ORIG" "$MXFP8" <<'PY'
import json, os, struct, sys

def head(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return {k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"}

uni, orig, mxfp8 = sys.argv[1:4]
tc = json.load(open(f"{uni}/config.json"))["text_config"]
want = {"num_key_value_heads": 6, "linear_num_key_heads": 24, "linear_num_value_heads": 72,
        "shared_expert_intermediate_size": 768, "moe_intermediate_size": 768,
        "vocab_size": 248320, "make_ngram_vocab_size_divisible_by": 384,
        "ple_embedding_dtype": "nvfp4"}
mal = {k: (tc.get(k), v) for k, v in want.items() if tc.get(k) != v}
if "tp3_orig_vocab_size" in tc:
    mal["tp3_orig_vocab_size"] = (tc["tp3_orig_vocab_size"], "fuera: el fork no lo conoce")
print("  config:", "OK " + str(want) if not mal else f"MAL {mal}")
wm = json.load(open(f"{uni}/model.safetensors.index.json"))["weight_map"]
swm = json.load(open(f"{orig}/model.safetensors.index.json"))["weight_map"]
ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
h = head(f"{uni}/{wm[ple + '.shard_0.weight']}")
g = head(f"{uni}/{wm[ple + '.weight_scale_2']}")[ple + ".weight_scale_2"]  # vive en otro fichero
print("  PLE shard_0:", h[ple + '.shard_0.weight']["dtype"], h[ple + '.shard_0.weight']["shape"],
      "| escala:", h[ple + '.shard_0.weight_scale']["dtype"], h[ple + '.shard_0.weight_scale']["shape"],
      "| global:", g["dtype"], g["shape"])
ho = head(f"{orig}/{swm[ple + '.shard_127.weight']}")
print("  en el ORIGINAL:", ho[ple + '.shard_127.weight']["dtype"], ho[ple + '.shard_127.weight']["shape"],
      "(las 256 filas de relleno van al final de la tabla global, no dentro de cada parte)")
for t in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
    print(f"  {t}:", head(f"{uni}/{wm[t]}")[t]["shape"], "(filas reales, sin recortar ni padrear)")
print("  tensores del index:", len(wm), "| igual que el ORIGINAL:", set(wm) == set(swm))
n = enl = 0
for f in sorted(os.listdir(uni)):
    if not f.endswith(".safetensors"):
        continue
    n += 1
    o = f"{orig}/{f}"
    if os.path.isfile(o) and os.stat(f"{uni}/{f}").st_ino == os.stat(o).st_ino:
        enl += 1
print(f"  ficheros: {n} shards, {enl} ENLAZADOS DUROS al hub (el resto, escritos nuevos)")
mh = head(f"{mxfp8}/model-00035-of-00036.safetensors")
k = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
print("  copia MXFP8", os.path.basename(mxfp8), ":", mh[k]["shape"],
      "(forma GLOBAL padreada: _tp_shard la corta por rango en 1, 2 y 3)")
PY
df -h "$CACHE" | tail -1
echo "== hecho $(date -Is): MODEL_DIR=$UNI  VLLM_B12X_NVFP4_MXFP8_CHECKPOINT=$MXFP8"
