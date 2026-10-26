# DGX-852 · vista UNIVERSAL: una sola versión del ORIGINAL con la PLE en NVFP4 (TP=1, 2 y 3)

Preparado 14-10-2026. Nada aplicado: ni vistas construidas, ni clúster tocado, ni GPU movida.

## 1. Camino elegido y por qué

**Camino A — modo `--universal` del conversor: una vista nueva, en NVFP4, derivada del checkpoint
ORIGINAL (el snapshot `16c9bd54` del hub), sin tocar el cargador del fork.** Se eligió porque el
cargador del fork ya sabe leer shards PLE en NVFP4 (`ple_layer.py::_STORAGE_MODES` → `nvfp4_group16`,
con `shard_N.weight` U8 + `shard_N.weight_scale` F8_E4M3 + `weight_scale_2` F32), de modo que lo
único que falta es **re-trocear la tabla al divisor que parte en 3 y rellenar la geometría**, que es
un trabajo de datos, no de código; el camino B (parchear el arranque para padrear MoE/PLE en el
momento de la carga) obligaría a reimplementar dentro del fork el mapeo de geometría completo
(réplica de KV, huecos GDN, conv1d, `A_log`/`dt_bias`, recorte del `out_proj`, re-troceo de la PLE
cruzando ficheros) sobre un binario que no poseemos y que se renueva con la imagen: más superficie,
más riesgo y ninguna garantía que la vista escrita no da ya.

**Con una reserva honesta (sección 5): la vista única NO cabe en TP=1.** El relleno que la hace
válida para TP=3 (expertos 640→768) son 13,6 GiB de pesos muertos en un solo rango, y un Spark con
121 GiB ya anda a 6 GiB libres con el ORIGINAL y los 14 GiB de KV de `serve-1x.sh`. En TP=2 y TP=3
la vista universal es exactamente lo pedido; en TP=1 la única versión que cabe es la sin relleno,
o sea el ORIGINAL (mismo formato NVFP4, misma PLE de 26,8 GiB).

## 2. Formato de la vista

`/home/dibanerz/.cache/huggingface/qwen38fn-mtp-ursu-universal`, construida en dgx1 con
`convert-ursu-nvfp4-tp3.py --universal` sobre el ORIGINAL. Cambios frente al ORIGINAL, y nada más:

| Cosa | ORIGINAL | vista universal | Por qué |
|---|---|---|---|
| PLE, dtype | `nvfp4` (U8 empaquetado 2/byte + escala e4m3 por grupo 16) | **igual: `nvfp4`** | es el punto del encargo: 90 bytes por fila (80 de peso + 10 de escala) en vez de los 160 de la vista FP8 |
| PLE, filas | 128 partes de 2.500.012 = 320.001.536 | 128 partes de **2.500.014** = **320.001.792** | `_contracts.py::storage_layout` exige `padded_vocab_size % tp == 0`; divisor 128→**384** |
| PLE, relleno | — | 256 filas **al FINAL de la tabla global**: peso `0x00`, escala `0x38` (=1,0 en e4m3) | `ple_layer.py::load_weights` rechaza escalas ≤ 0; `check` exige cobertura exacta por parte |
| `weight_scale_2` | F32 [1] | **igual** (sin renombrar) | la vista FP8 lo convertía en `weight_scale` BF16; aquí no hay conversión |
| KV heads | 2 | **6** | 6 divide entre 1, 2 y 3 |
| GDN k:v | 16:48 | **24:72** | idem (8:24 por rango en TP=3, 12:36 en TP=2) |
| shared expert | 640 | **768** | múltiplo de 3·128 |
| expertos enrutados | 640 | **768** | `b12x.py:495` rechaza EP y sin EP 640 no se parte en 3 |
| `vocab_size`, `embed_tokens`, `lm_head` | 248320 | **248320 intacto** | de `vocab_size` salen los multiplicadores del hash PLE (`ple_hash/reference.py::ple_multipliers`, comparados con `torch.equal`); el pad a `lcm(64,tp)` lo hace el fork solo |
| `tp3_orig_vocab_size` | no existe | **no se escribe** | invento nuestro que el fork no conoce |
| `hf_quant_config.json`, tokenizer, etc. | — | copiados/enlazados tal cual | el fichero no lleva dimensiones, solo `quant_algo`/`group_size` por nombre |

Lo que **no** cambia de bytes se **enlaza duro** contra los blobs del hub (`place()` → `os.link`, que
sigue el symlink del snapshot hasta el blob; mismo inodo, cero duplicado). Solo se puede hacer como
root: los blobs son `root:root` y con `fs.protected_hardlinks=1` enlazar lo ajeno da EPERM, y
`place()` aborta en vez de copiar a ciegas.

## 3. Tamaño en disco y por rango en cada TP

Medido con `check --universal` sobre el checkpoint REAL de dgx1 (lee solo cabeceras):

```
vista: 119.445.280.392 reescritos + 9.669.968 copiados (1 shard) = 111,3 GiB
relleno: 151337 tensores {'ab':144,'conv':36,'expert_down':50176,'expert_in':100352,'kv':50,
        'out':72,'qkv':72,'shared_down':97,'shared_in':194,'vhead':72,'z':72}
universal model.language_model.layers.1: 128 partes de 2500012 -> 2500014 filas
(tabla 320.001.536 -> 320.001.792, relleno 256 al final, divisor 384); reescribe 17 ficheros
config.json (text_config): {'num_key_value_heads': 6, 'linear_num_key_heads': 24,
 'linear_num_value_heads': 72, 'shared_expert_intermediate_size': 768, 'moe_intermediate_size': 768,
 'vocab_size': 248320, 'make_ngram_vocab_size_divisible_by': 384, 'ple_embedding_dtype': 'nvfp4'}
```

| | ORIGINAL | vista universal | vista FP8 de antes (`--fork-view`) |
|---|---|---|---|
| ficheros en disco | 104.866.295.752 B = **97,7 GiB** | 119.454.950.360 B = **111,3 GiB** | ~132 GiB (los mismos + PLE en FP8) |
| PLE (1 capa, `ple_layer_ids:[2]`) | 26,8 GiB NVFP4 | **26,8 GiB NVFP4** | 47,7 GiB FP8 |
| ficheros reescritos / enlazados | — | **35 / 1** | 17 / 19 |
| pesos por rango TP=3 | — (no carga: 640 y 320.001.536 no parten en 3) | **37,1 GiB** | 44,0 GiB |
| pesos por rango TP=2 | 48,8 GiB | **55,6 GiB** | — |
| pesos por rango TP=1 | 97,7 GiB | **111,3 GiB** | — |

## 4. Coste del relleno en TP=1 y TP=2 (lo que se paga por la vista única)

Todo el coste es relleno **muerto** (filas/columnas a cero que en 1 y 2 sobran: 640 ya partía en 2
y en 1):

* **En disco / residente: +13,59 GiB (14.588.654.608 B) sobre el ORIGINAL**, igual en TP=1, TP=2 y
  TP=3 porque es el mismo fichero.
* **Por rango: TP=1 +13,6 GiB, TP=2 +6,8 GiB, TP=3 +4,5 GiB.**
* Desglose del relleno (derivado del config real y anclado contra las cabeceras de dgx1):
  expertos `640→768` = 552.960 B/experto/capa × 512 × 49 capas (48 + MTP) = **12,9 GiB**;
  shared expert `640→768` ≈ 0,02 GiB; KV `2→6`, GDN `16:48→24:72` y `out_proj` por columnas
  ≈ 0,3 GiB; espejo MXFP8 interno (shard 35, 588 tensores, sin expertos) +0,6 GiB; PLE +23 KiB
  (256 filas × 90 B).
* **Lo que se gana:** frente a la vista FP8 son **20,9 GiB menos** de PLE (26,8 vs 47,7) y deja de
  hacer la cadena de tres pasadas (original → vista FP8 → e768 → fork-view) por una sola pasada.

## 5. Riesgos

1. **TP=1 no cabe (duro).** 111,3 GiB de pesos + 14 GiB de KV (`--kv-cache-memory-bytes
   15032385536` de `serve-1x.sh`) = 125,3 GiB sobre los 121 GiB del Spark; hoy, con el ORIGINAL,
   `free -g` en dgx3 marca **6 GiB disponibles de 121** con el servidor levantado
   (`mtp/definitivo/tp21-A/uno.log`, 11:32Z). La vista universal en TP=1 solo entraría bajando el
   KV por debajo de ~1 GiB, que no sirve para 262k de contexto. `serve-1x-universal.sh` lo avisa en
   el preflight; la decisión (bajar KV, o seguir sirviendo el ORIGINAL en TP=1) es de quien lanza.
2. **Cabezas GDN/KV muertas en TP=1 y TP=2.** La vista trae 24 cabezas clave GDN (16 reales) y 6 de
   KV (2 reales): en TP=1 los 8 grupos sobrantes viven en el mismo rango, con pesos a 0, `A_log`/
   `dt_bias` a 0 y estado que se queda en 0. El mecanismo está probado en TP=3 (esa es la forma que
   ya carga), pero **no está medido en GPU en 1 y 2**: hay que correr la greedy de siempre contra la
   base del ORIGINAL antes de dar la vista por buena. Si algo divide por el número de cabezas reales
   o normaliza por un estado de cabeza nula, sale aquí.
3. **Formato NVFP4 en el cargador.** Es el formato que ursu publica, así que su ruta `nvfp4_group16`
   es la suya principal, no la nuestra; pero nosotros no la habíamos ejercitado (nuestra pila solo
   sabe FP8). El preflight de las recetas comprueba dtype y formas antes de lanzar.
4. **Disco.** Escribir la vista son **111,3 GiB nuevos** (35 ficheros; solo 1 queda enlazado). Con
   los números de hoy en dgx1 (348,1 GB libres de 982,8 GB, 64,6% de uso) el pico con los `.tmp`
   deja **220,7 GB libres y 77% de uso**: por debajo del 80% pedido pero con solo ~32 GB de margen.
   `construir-universal.sh` calcula la cifra con `check` (no con un literal) y aborta si se pasa del
   techo o si no queda un 20% del volumen (el `eviction-hard` del kubelet es 10%: el incidente del
   §9.2 de `../fork-tp3/INFORME.md`). Si se queda corto, lo que sobra es la cadena vieja
   (`qwen38fn-mtp-ursu-nvfp4-tp3`, `-e768`, `-tp3-fork`: ~350 GiB entre las tres).
5. **Copia MXFP8: se reutiliza, no se rehace.** `qwen38fn-mtp-ursu-mxfp8-tp3` guarda las formas
   **globales** padreadas (`in_proj_a` [72,2560], `in_proj_qkv` [15360,2560]) y `_tp_shard` la corta
   por rango, así que vale para 1, 2 y 3. Comprobado en dgx1: su shard 35 tiene 480 tensores y **0 de
   expertos**, así que el relleno 640→768 no le afecta. Si el directorio no existe,
   `construir-universal.sh` lo construye con `--mxfp8-copy`.
6. **`--universal` es excluyente** (`--fork-view`, `--moe-pad`, `--solo-ple`, `--mxfp8-copy`,
   `--only`): necesita ver todos los tensores para calcular la geometría y exige que la fuente sea
   el ORIGINAL en NVFP4 (aborta con la fuente en FP8, con partes que ya parten en 3, con peso y
   escala de una parte en ficheros distintos, con expertos ya múltiplos de 384 o con el config en
   otro dtype).

## 6. Ficheros

| Fichero | Qué es |
|---|---|
| `mtp/checkpoint/convert-ursu-nvfp4-tp3.py` | modo `--universal` (plan, re-troceo NVFP4, config, verify) |
| `mtp/checkpoint/test/test_universal.py` | prueba en CPU (números reales + miniatura byte a byte + fail-closed) |
| `mtp/definitivo/universal/construir-universal.sh` | construcción en dgx1 con guarda de disco |
| `mtp/definitivo/fork-tp3/serve-3x.sh` | preflight actualizado: acepta la vista universal (NVFP4) y la vieja FP8; `MODEL_DIR` por defecto a la universal |
| `ops/recipe/serve-2x-universal.sh`, `ops/recipe/serve-1x-universal.sh` | variantes que sirven la vista **por ruta** (sin `--revision`), con preflight por TP |

## 7. Comandos exactos

Prueba en CPU (aquí, cero GPU, cero red — la miniatura y los números reales):

```bash
cd /home/user/bench-results/dgx-847/mtp/checkpoint/test
python3 test_universal.py      # la vista universal
python3 test_cpu.py && python3 test_fork_view.py && python3 test_e768.py   # regresión de los otros modos
```

Construir la vista (dgx1; hay que llevar el conversor nuevo y su `patches/` a `/home/dibanerz/dgx-852/`,
que el script comprueba antes de mover un byte):

```bash
ssh dgx 'sudo bash /home/dibanerz/dgx-852/construir-universal.sh'          # aborta si la vista ya existe
ssh dgx 'sudo bash /home/dibanerz/dgx-852/construir-universal.sh --fuerza' # y reconstruye
# solo el plan y la cifra de disco, sin escribir:
ssh dgx 'python3 /home/dibanerz/dgx-852/correr.py check --src <snapshot ORIGINAL> --universal'
```

Servir (variables de entorno **nuevas**: `MODEL_DIR` y `MXFP8_DIR`; la copia MXFP8 se pasa con
`VLLM_B12X_NVFP4_MXFP8_CHECKPOINT`, que las variantes exportan solas porque `env-1x.env`/`env-2x.env`
siguen apuntando al hub):

```bash
# TP=3 (receta de siempre, ahora con la vista universal por defecto)
MODEL_DIR=/cache/huggingface/qwen38fn-mtp-ursu-universal \
MXFP8_DIR=/cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3 bash serve-3x.sh
# TP=2 y TP=1 (nuevas): por ruta, sin id de HF ni revisión
MODEL_DIR=... MXFP8_DIR=... bash ops/recipe/serve-2x-universal.sh
MODEL_DIR=... MXFP8_DIR=... bash ops/recipe/serve-1x-universal.sh
```

Quien monte las recetas en el pod (`ops/ventana.sh:35`, ConfigMap `ursu-recipe` con `--from-file`
explícito por fichero) tiene que añadir los dos nombres nuevos; no lo he tocado.

## 8. Resultado de las pruebas de CPU (14-10, esta sesión)

```
1) números REALES del checkpoint ORIGINAL (config sin empequeñecer)
   PLE: 2.500.012 x 128 = 320.001.536 filas -> padded 320.001.792 (divisor 128->384),
        partes de 2.500.014, relleno 256
   geometría: KV 2->6, GDN 16:48->24:72, shared 640->768, expertos 640->768
   PLE por capa: NVFP4 26.8 GiB (80+10 bytes/fila) frente a FP8 47.7 GiB -> se guardan 20.9 GiB
   relleno de expertos: 552.960 bytes/experto/capa x 512 x 49 capas = 12.9 GiB
   familia real 0.mlp.shared_expert.gate_proj: F8_E4M3 -> peso /1, escala /32 [640,2560] (MXFP8)
   familia real 3.self_attn.k_proj:              F8_E4M3 -> [512,2560]; paso 128 sin partir grupos
2) miniatura 4 partes de 5 filas (divisor 4) -> 4 de 6, moe 16->384, KV 2->6
3) re-troceo NVFP4 bit a byte contra un camino ingenuo: 34 filas comparadas, 0 distintas;
   relleno 0x00 en peso y 0x38 en escala, SOLO al final de la tabla global; weight_scale_2 F32 intacto
4) 14 tensores de geometría: gather y relleno idénticos al camino escrito a mano
5) vocabulario 1000 filas intacto y ENLAZADO DURO; config: cambian exactamente 6 claves
   (kv, gdn k/v, shared, moe, divisor); `vocab_size` y `ple_embedding_dtype: nvfp4` igual
6) verify pasa=True errores=0; divisibilidad 1/2/3 comprobada sobre el config escrito
7) fail-closed: peso en FP8, parte ausente, partes que ya parten, peso y escala en ficheros
   distintos, expertos ya padreados, config en FP8 y `--only` -> los seis abortan con su mensaje
OK (y sin regresión: test_cpu, test_fork_view y test_e768 también pasan)
```
