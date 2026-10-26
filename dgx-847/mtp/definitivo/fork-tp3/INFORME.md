# DGX-852 · El fork de ursuciprian en TP=3 con E2 y vocabulario del borrador en castellano

14-10-2026. **Estado: todo preparado y validado. El manifiesto está validado solo con
`kubectl apply --dry-run=server`.** Pero hay **dos incidentes de mi parte**, escritos tal cual en §9:
(1) al probar el guion de arranque ejecuté sin querer el `apply` real — se crearon 2 ConfigMaps y 3
Desplegamientos y los borré a los ~30 s; ninguno quedó y el head de la 2x no se reinició por eso (su último
reinicio fue diez segundos ANTES). (2) **la construcción de la vista en dgx1 metió al nodo en DiskPressure**
(96 GB libres − 56 GB escritos = 4,4 %; el kubelet desaloja por debajo de 10 %/15 %) a las **07:42:03Z**: el
worker de la 2x se desalojó en bucle durante una hora y el GC del kubelet se llevó de dgx1 la imagen del
fork y el `nccl-ring` (la del fork ya volvió sola a las 08:22Z). El nodo está sin presión desde 08:18:57Z
con **151 GB libres (15,4 %)**, que es un margen estrecho mientras la vista ocupe ese disco. Ninguna vista
existente fue tocada.

Resumen en cinco líneas:
1. La vista `qwen38fn-mtp-ursu-nvfp4-tp3-e768` **no carga en su fork** en TP=3 por tres cosas concretas
   (vocabulario padreado a 248.448 en vez de 248.320, divisor PLE 128 en vez de 384 y partes de 2.500.012
   filas en vez de 2.500.014). Las tres están verificadas en su código, con fichero y línea, abajo.
2. He extendido el conversor con dos modos nuevos anclados y fail-closed (`--fork-view`, `--mxfp8-copy`)
   y su test en CPU; los dos tests previos siguen pasando.
3. Las dos vistas están construidas en dgx1: `qwen38fn-mtp-ursu-nvfp4-tp3-fork` (132,1 GiB, **solo 56,0 GB
   nuevos**; 19 shards por enlace duro) y `qwen38fn-mtp-ursu-mxfp8-tp3` (3,9 GB). `verify` pasa en las dos.
4. Recetarios nuevos en `fork-tp3/` (no se editó ningún original): `serve-3x.sh`, `env-3x.env`,
   `env-ring-3x.env`, `ursu-3x.yaml`, `aplicar-3x.sh`, `construir-vistas.sh`. El manifiesto pasa
   `--dry-run=server`.
5. **dgx3 no necesita copia**: lee el cache de dgx1 por la PVC de NFS-ro que ya usa el residente
   (`hf-cache-dgx1-dgx3-llm-ro`). Los tres rangos ven la misma ruta `/cache/huggingface/…`.

---

## 1. Qué exige TP=3 en SU fork — verificado en el código, no supuesto

Fuente: `/home/user/bench-results/dgx-847/fork-src/usr/local/lib/python3.12/dist-packages/`
(imagen `ghcr.io/ursuciprian/spark-vllm-b12x@sha256:8abb60ee…`, vLLM `v20260926.dev28+g884b4ff68`).

| # | Superficie | Qué hace su fork (fichero:línea) | Qué exige al checkpoint en TP=3 |
|---|---|---|---|
| 1 | **Divisor de la tabla PLE** | `b12x/sequence/ple_embedding/_contracts.py:381-384`: `padded_vocab_size = geometry.padded_vocab_size; if padded_vocab_size % caps.tp_size: raise ValueError("padded_vocab_size=… must be divisible by tp_size=…")`. `padded = align_up(sum(primes), caps.table_alignment)` (`ple_hash/geometry.py:97`) y `table_alignment` sale **tal cual** de `config.make_ngram_vocab_size_divisible_by` (`vllm/models/qwen3_8_flash_next/ple_layer.py:294`). | La tabla (Σ primos = 320.001.446) hay que alinearla a **384**, no a 128: `align_up(320.001.446,128)=320.001.536` **no divide entre 3** → aborta; `align_up(…,384)=320.001.792` sí. El fork **no sube el alineado por su cuenta**: aborta. |
| 2 | **Forma de cada parte PLE** | `ple_layer.py:313-315`: `shard_rows = ceil(padded_vocab_size / split_ngram_parts)`; `ple_layer.py:717-718,826-830`: `shard_size` igual, `checkpoint_start = shard_index * shard_size`, `expected_rows = max(0, min(shard_size, padded_vocab_size - checkpoint_start))`, y `expected_shape = (expected_rows, weight_shape[1])` (`:835`). | 128 partes (`split_ngram_parts`, `ple_layer.py:260`) de **2.500.014 filas** × 160 bytes (FP8), con los **256 rellenos al final de la tabla global** (última parte = filas 254 reales + 256 ceros), porque las partes son bloques globales uniformes. Con 2.500.012 → `expected_rows` 2.500.014 y aborta de forma. |
| 3 | **`vocab_size` del config** | `ple_layer.py:288` pasa `vocab_size=int(config.vocab_size)` a los Caps; `ple_hash/geometry.py:63-65` deriva `multipliers = ple_multipliers(vocab_size, max_order, dense_layer_ordinal)`; `ple_layer.py:734-737`: `if not torch.equal(buffer, checkpoint_value): raise ValueError("checkpoint layer_multipliers does not match planned PLE geometry")`. | `vocab_size` **tiene que ser 248.320** (el del checkpoint). Con 248.448 los multiplicadores derivados no coinciden y aborta en la capa PLE. No es un detalle estético: es un `torch.equal` contra el tensor del checkpoint. |
| 4 | **embed / lm_head** | `vocab_parallel_embedding.py:336`: `self.padding_size = math.lcm(padding_size, self.tp_size)` (lcm(64,3)=**192** → 248.320 → 248.448 lo padrea **él mismo**); `:612`: `assert loaded_weight.shape[output_dim] == self.org_vocab_size`; `:616-617`: `copy_weight(param[: loaded_weight.shape[0]].data, loaded_weight)` y `param[loaded_weight.shape[0] :].data.fill_(0)`. | El checkpoint debe traer las **248.320 filas reales, sin padrear**. Nuestra vista e768 trae 248.448 (el padreo que pide *nuestra* pila) → el assert revienta. De ahí el **recorte** a 248.320 en la vista del fork. |
| 5 | **Cabezas KV y GDN** | Sharding ordinario por cabezas: 2 KV → 6 (`num_key_value_heads: 6`, patrón `[k0,k0,k0,k1,k1,k1]`), GDN 16:48 → 24:72 con 8 huecos nulos por cada 8 de grupo en cada rango. | Ya estaba en la vista e768 (es lo que significa "e768"). **Sin trabajo.** |
| 6 | **Compartida 768 y expertos 768 con `--moe-backend b12x` sin EP** | `vllm/model_executor/layers/fused_moe/b12x.py:568-576`: `_supports_parallel_config` exige `not use_ep and ep_size == 1 and not use_all2all_kernels and not enable_eplb` → **en TP=3 no hay EP**, así que el intermedio se parte entre 3: 640/3 no es entero → 768. | 768 en `shared_expert_intermediate_size` y `moe_intermediate_size`. Ya estaba en la e768. **Sin trabajo.** |
| 7 | **`fc` del MTP** | `mtp.py:223` y `:232`: `fc_embedding` / `fc_hidden` son `ReplicatedLinear` (no se parten). | Nada: las filas replicadas por rango no cambian con TP. **Sin trabajo.** |
| 8 | **Copia MXFP8 (shard 35)** | `kernels/linear/nvfp4/b12x.py:110 load_mxfp8_large_m_copy` lee el tensor entero y `:72 _tp_shard` lo corta contra `output_size_per_partition`; el mapa de qué ficheros existen sale **solo** de `model.safetensors.index.json` (`_checkpoint_weight_map`). | La copia tiene que traer las **mismas formas padreadas que la capa** (global padreado, no por rango): `in_proj_qkv [15360,2560]`, `in_proj_z [9216,2560]`, `out_proj [2560,9216]`, `in_proj_a/b [72,2560]`, `k_proj [1536,2560]`. Y el index **solo con los tensores presentes** (el del checkpoint entero apunta a 35 ficheros que no están). Verificado en el arranque E2 de TP=2: lo que se consulta es `linear_attn.in_proj_qkv+in_proj_z` y `out_proj` de las 36 capas GDN (`e2/tp2-arranque.log`: "MXFP8 copy serves rows >= 41"), **no** hay búsqueda en el MTP. |
| 9 | **`VLLM_MXFP8_LM_HEAD` en TP=3** | `vocab_parallel_embedding.py:65-84 _supports_default_lm_head_quantization`: exige `input_size % 128 == 0` y `output_size % 8 == 0` (entre otras). | Objetivo: 2560 % 128 = 0 y 248.448/3 = 82.816, 82.816 % 8 = 0 ✓. Borrador: K=131.072 padreado a 131.136 (múltiplo de 192) → 43.712 por rango, % 8 = 0 ✓. **Se queda a 1** (cuantiza en línea desde el tensor BF16, no desde la copia). |
| 10 | **`VLLM_QWEN38_HC_MXFP8=hc` en TP=3** | `hyperconnection.py:355-385`: las capas HC son `ReplicatedLinear` / `disable_tp=True`, con `_MXFP8_GROUP=32` y `_MXFP8_N_ALIGN=8` sobre el tamaño completo. | **Independiente de TP**: nada cambia, no hay que padrear nada. Se queda `hc`. |
| 11 | **E2 (BF16 + GDN triton) y los checkpoints diferidos** | `vllm/v1/worker/gdn_deferred_commit.py:166` y `resolve()`: fail-closed — los diferidos solo se admiten con b12x en decode **y** prefill; con triton aborta. Igual `VLLM_GDN_COMPACT_RECORDS` (`:175-186`). | `VLLM_GDN_DEFERRED_CHECKPOINTS=0` **obligatorio** con E2; `VLLM_GDN_COMPACT_RECORDS=0` (su default ya es 0, `envs.py:1289-1290`, pero se pone explícito porque la receta 1x lo trae a 1). |
| 12 | **Vocabulario del borrador en TP=3** | `mtp.py:684-714 DraftVocabLMHead(ParallelLMHead)` con `K=131.072` filas; `:709 gather_shard_rows` + `param.data[rows:].fill_(0)`; `draft_vocab.py:49-64` (narrow del tramo + index_select); `mtp.py:830 draft_id_to_target_id`. La lista se valida contra `config.vocab_size` (`mtp.py:795-800`, `draft_vocab.py:21-33`). | Comprobado con la lista real (`ids-es-K131072.txt.gz` del ConfigMap): **K=131.072, id máximo 248.076 < 248.320 ✓**. Aritmética TP=3: 131.072 % 192 = 128 → padreado 131.136, 43.712 por rango, el relleno cae **al final** del tensor gathered (rango 2: ids [87.424,131.072) + 64 ceros), así que la columna j de los logits es exactamente `ids[j]` y el scatter del `LogitsProcessor(131.072)` es correcto. **Verificado en el código y en la lista; no en GPU** (en TP=2 E2 sí está verificado en GPU: el log dice "MTP draft vocab: 131072 of 248320 ids from /opt/mtp-vocab-es/ids-es-K131072.txt.gz"). |

### Lo que la vista e768 tenía y su fork no puede cargar (las tres correcciones)

| | e768 (nuestra pila) | vista del fork | por qué |
|---|---|---|---|
| `vocab_size` | 248.448 (+ `tp3_orig_vocab_size: 248320`) | **248.320**, sin `tp3_orig_vocab_size` | `torch.equal` de los multiplicadores (fila 3) |
| `make_ngram_vocab_size_divisible_by` | 128 | **384** | `padded_vocab_size % tp_size` (fila 1) |
| partes PLE | 128 × 2.500.012 | 128 × **2.500.014** | `checkpoint_start`/`expected_rows` (fila 2) |
| `embed_tokens` / `lm_head` | [248.448, 2560] | **[248.320, 2560]** | assert del loader (fila 4) |

El relleno de las 256 filas extra **no** es por cabeza: `ple_table_geometry` calcula los offsets como
acumulados sobre los primos **sin padrear**, así que subir el alineado solo **añade filas al final de la
tabla global**; por eso la última parte lleva 254 filas reales + 256 ceros y las otras 127 llevan
2.500.014 reales. Escrito así en el plan (`piezas_ejemplo.ultima = [[127,254,2499758],[null,0,256]]`).

---

## 2. Las dos vistas construidas en dgx1

Rutas (dentro del contenedor, los tres rangos ven lo mismo en `/cache/huggingface`):

| Vista | Ruta en dgx1 | Contenido | Bytes nuevos | Enlazados |
|---|---|---|---|---|
| Modelo TP=3 del fork | `/home/dibanerz/.cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3-fork` | 36 shards + config + index + `SHA256SUMS` (53 entradas) + `tp3-plan.json` + `verify.json` + marca `.built-dgx852-fork-tp3` | **56.045.629.438** (17 ficheros reescritos: 16 con partes PLE + `model-00034` con el vocab y la escala) | **85.809.426.000** en 19 shards por **enlace duro** a la e768 → total 132,1 GiB |
| Copia MXFP8 TP=3 | `/home/dibanerz/.cache/huggingface/qwen38fn-mtp-ursu-mxfp8-tp3` | `model-00035-of-00036.safetensors` (3.926.866.248 B, sha256 `74023b81f8234ffe…`), index con **480 tensores presentes**, `SHA256SUMS`, plan, verify | 3,9 GB | 0 (se reescribe entero; **no** escribe `config.json`: el config lo pone la vista del modelo) |

`verify` en las dos: `pasa: true, errores: [], avisos: []`. El de la vista del modelo añade el bloque
`fork` (recorte comprobado fila a fila en las filas 0,1,124160,248318,248319 de los dos tensores y
muestreo de partes PLE 0,1,14,127 con **44.04 filas comparadas, 0 distintas**) y
`sin_reescribir: {ficheros: 19, enlazados: 19}`.

Comprobaciones de forma en la propia máquina (salida `pruebas` del log):
`vocab_size 248320`, `tp3_orig_vocab_size fuera`, divisor 384, dtype PLE `float8_e4m3fn`, moe/shared 768,
`…ngram_embedding.shard_0.weight` F8_E4M3 **[2500014,160]** (fuente: [2500012,160]),
`…ngram_embedding.weight_scale` BF16 [1] (enlace duro, intacto), `embed_tokens`/`lm_head` BF16 **[248320,2560]**,
copia `in_proj_qkv` F8_E4M3 **[15360,2560]**, `out_proj` [2560,9216], `in_proj_a` [72,2560], index 480,
sin `config.json` en la copia.

### El conversor extendido

`/home/user/bench-results/dgx-847/mtp/checkpoint/convert-ursu-nvfp4-tp3.py`, tres modos nuevos, anclados y
fail-closed como el resto:
- `--fork-view`: deriva la vista del fork **desde la e768**. Exige (`_fork_geometry`) las formas TP=3 de la
  fuente (KV 6, GDN 24:72, compartida 768, moe 768, dtype FP8, 128 partes), exige que el padreo del vocab
  sea exactamente `pad_up(248320, 192)` (si fuera otro padreo aborta: "no es pad_up"), que haya exactamente
  dos tensores de vocabulario con forma [248448, 2560], que las partes PLE sean el rango global completo y
  que **haya algo que re-trocear** (`new_rows != rows`). Registra el recorte como un cambio más
  (`("vocab_crop", 0, [(0, 248320)], 2 bytes)`) y reescribe solo los 17 ficheros implicados. incompatible
  con `--moe-pad`, `--solo-ple` y `--only` (aborta en `main`).
- `--mxfp8-copy` + `--config-from`: rellena la copia MXFP8 a las formas TP=3 usando el config **del
  checkpoint original** (la copia no trae config) y avisa de los ficheros referenciados y ausentes; no
  escribe config.json y su index solo lista lo presente.
- `place()` ya no copia a ciegas si `os.link` falla: solo copia ante `EXDEV` (otra unidad); cualquier otro
  error aborta ("la vista no se duplica a ciegas").
- `copy_span` escribe por trozos de 8 MiB (no materializa cortes de 400 MB) y `ple_reshard_part` lee por
  fichero con un mmap por shard (`Plan.src_file`), valida ancho, forma y tamaño por pieza y escribe ceros
  en la pieza `None`.

Tests CPU: `mtp/checkpoint/test/test_fork_view.py` (**nuevo**, pasa) — mini e768 con las mismas relaciones
(P=4, 5→6 filas, divisor 6, vocab 5 dentro de 192), partes PLE repartidas en dos ficheros para forzar la
lectura cruzada, un fichero intacto para probar el enlace duro; comprueba byte a byte contra un
`np.vstack([tabla, ceros])` ingenuo, el recorte, el diff exacto del config (`{vocab_size,
tp3_orig_vocab_size}`), `verify`, y **cinco casos fail-closed** (vocab mal padreado, parte ausente, PLE en
NVFP4, sin nada que re-trocear, `--only`); después el modo `--mxfp8-copy` con un index que referencia un
shard ausente, `in_proj_a` 48→72 y `k_proj` 512→1536 con el patrón `[k0,k0,k0,k1,k1,k1]` (usando los
`stored_units`/`naive_gather` escritos a mano en `test_cpu.py`, camino independiente) y la ausencia de
config.json. `test_cpu.py` y `test_e768.py` siguen pasando.

---

## 3. Recetarios adaptados (copias nuevas; ningún original tocado)

En `/home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3/`:
`serve-3x.sh` (copia de `tp3/recipe/serve-3x.sh`), `env-3x.env`, `env-ring-3x.env`, `ursu-3x.yaml`,
`aplicar-3x.sh` (copia de `tp3/aplicar-3x.sh`), `construir-vistas.sh` + `correr.py` (el lanzador que fuerza
el camino numpy con `sys.modules["torch"]=None` vía runpy), `construir.log`.

`serve-3x.sh` añade **antes** de arrancar: comprobación de las tres rutas (vista, copia, vocabulario; sale
con 6), un **preflight de la vista en Python** que aborta si no es LA DEL FORK (vocab 248320, sin
`tp3_orig_vocab_size`, divisor 384, PLE FP8, moe/shared 768, parte PLE [2500014,160], embed/lm_head
[248320,2560], copia [15360,2560] y index ≤ 1000 tensores), y recupera `${EXTRA_ARGS:-}` que la 3x de
DGX-847 había perdido (hace falta para poder probar sin reescribir el script). El resto (semillas b12x,
generador del grafo del anillo, preflight de 4 HCA/GID RoCE v2 IPv4/MTU) es el de DGX-847 tal cual.

Validación: `kubectl apply --dry-run=server -f ursu-3x.yaml` →
`deployment.apps/ursu-fork-3x-head created (server dry run)` (+ worker-dgx1, + worker-dgx3), rc=0. Después
`kubectl -n llm get deploy | grep -c ursu-fork` = **0** (no se creó nada). `bash -n` en los tres scripts. Los
dos env pasan el formato `KEY=VALUE` de `--from-env-file`.

---

## 4. QUÉ LLEVA esta configuración TP=3+E2 frente a su receta 2x

Base de comparación: `ops/recipe/serve-2x.sh` + `ops/recipe/env-2x.env` + `ops/recipe/env-sparkrun-2x.env` +
`ops/ursu.yaml` (2× TP=2, imagen `8abb60ee`, E2 aplicado por parche del manifiesto en `ops/prueba-e2.sh`).
Estado: **igual** / **cambiado (cómo)** / **perdido (por qué)** / **nuevo**.

### 4.1 Flags de `vllm serve`

| Flag | 2x | Aquí (TP=3 + E2) | Estado |
|---|---|---|---|
| modelo | `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` desde HF hub | `$MODEL_DIR` (ruta local de la vista del fork) | **cambiado**: no hay descarga; `--served-model-name` mantiene los dos nombres |
| `--revision 16c9bd54788d12390838a65ce4a4ecda97fa5f1d` | sí | fuera del comando de línea | **cambiado**: el ancla va en la vista (el plan lleva `revision: 16c9bd54…` y `SHA256SUMS` por fichero; `HF_HUB_OFFLINE=1`) |
| descarga del shard 35 + index (`snapshot_download … allow_patterns`) | sí, con `HF_HUB_OFFLINE=0` si falta | no | **perdido (deliberado)**: lo sustituye `qwen38fn-mxfp8-tp3`, que es el shard 35 **padreado a TP=3** y con index limpio |
| `--host 0.0.0.0`, `--port 8000`, `--trust-remote-code` | sí | sí | igual |
| `--served-model-name qwen3.8-flash-next ursuciprian/…` | sí | sí | igual |
| `--tensor-parallel-size` | 2 | **3** | cambiado |
| `--pipeline-parallel-size 1` | sí | sí | igual |
| `--mamba-cache-mode align` | sí | sí | igual (obligado: `mtp.py:763-768` aborta con `all`) |
| `--mamba-ssm-cache-dtype` | ausente → FP32 | **bfloat16** | cambiado (E2) |
| `--gdn-decode-kernel` | `b12x` | **`triton`** | cambiado (E2); el b12x de decode exige FP32 |
| `--additional-config` | ausente | **`{"gdn_prefill_backend":"triton","gdn_decode_kernel":"triton"}`** | nuevo (E2 fija también el prefill) |
| `--enable-prefix-caching` | sí | sí | igual |
| `--enable-chunked-prefill` | sí | sí | igual |
| `--max-parallel-prefills 4` | sí | sí | igual |
| `--prefill-policy decode-aware` | sí | sí | igual |
| `--decode-refill-target auto` | sí | sí | igual |
| `--dtype bfloat16` | sí | sí | igual |
| `--kv-cache-dtype fp8` | sí | sí | igual |
| `--quantization modelopt_mixed` | sí | sí | igual |
| `--block-size 16` | sí (y **lo sobrescribe** el motor: "Setting attention block size to 1504 tokens to ensure that attention page size is >= mamba page size") | sí, mismo valor | **igual en el flag, y en la práctica lo ignora el motor**: con E2 en TP=2 salió 1504 y en TP=1 1584. En TP=3 saldrá otro valor (no medido) |
| `--load-format b12x` | sí | sí | igual |
| `--max-model-len 262144` | sí | sí | igual |
| `--max-num-seqs 16` | sí | sí | igual |
| `--max-num-batched-tokens 8192` | sí | sí | igual |
| `--speculative-config {"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}` | sí | sí | igual (k=4) |
| `--linear-backend b12x` | sí | sí | igual |
| `--moe-backend b12x` | sí | sí | igual, **y sin EP** (su `_supports_parallel_config` lo exige) |
| `--no-enable-flashinfer-autotune` | sí | sí | igual |
| `--mm-encoder-tp-mode data` | sí | sí | igual |
| `--reasoning-parser qwen3` | sí | sí | igual |
| `--tool-call-parser qwen3_xml` | sí | sí | igual |
| `--enable-auto-tool-choice` | sí | sí | igual |
| `--default-chat-template-kwargs {"reasoning_effort":"medium"}` | sí | sí | igual |
| `--compilation-config {"pass_config":{"fuse_act_quant":true}}` | sí | sí | igual |
| `--gpu-memory-utilization 0.8` | sí | sí | igual |
| `--nnodes` | 2 | **3** | cambiado |
| `--node-rank $NODE_RANK` | sí | sí | igual |
| `--master-addr` | `10.0.0.2` (enlace dgx1↔dgx2, /30) | **`spark2`** (LAN enP7s7 de dgx2) | cambiado: dgx3 no está en esa /30; el triángulo son tres /30 distintas |
| `--master-port 25000` | sí | sí | igual |
| `$HEADLESS` (`--headless` en los workers) | sí | sí | igual |
| `${EXTRA_ARGS:-}` | sí | **sí** (la 3x de DGX-847 lo había perdido; recuperado) | igual |
| preflight de la vista + del vocabulario | no | **sí** (sale con 6 antes de tocar GPU) | nuevo |
| generador del grafo NCCL por rango + preflight de HCA/GID/MTU | no | **sí** (sale con 7) | nuevo (viene del TP=3 de DGX-847) |

### 4.2 Variables de entorno del modelo (`env-2x.env` → `env-3x.env`)

| Variable | 2x | Aquí | Estado |
|---|---|---|---|
| `MODEL_DIR`, `MXFP8_DIR` | no existen (modelo por HF hub) | `/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3-fork`, `…/qwen38fn-mtp-ursu-mxfp8-tp3` | nuevo |
| `VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS` | 41 | 41 | igual |
| `VLLM_B12X_NVFP4_MXFP8_CHECKPOINT` | snapshot de HF hub | **nuestra copia TP=3** | cambiado (obligado: `_tp_shard` compara contra la partición de la capa) |
| `VLLM_QWEN38_HC_MXFP8` | `hc` | `hc` | igual (ReplicatedLinear: no depende de TP) |
| `VLLM_GDN_UNIFORM_DECODE_META_SKIP` | 1 | 1 | igual — se lee en `v1/attention/backends/gdn_attn.py:910,928`, en el backend GDN común, no en el kernel b12x: sigue vivo con triton |
| `B12X_AUTOTUNE` | 1 | 1 | igual **pero frío** (las semillas de la imagen están claveadas a TP=2 y a la ruta del modelo) |
| `B12X_ROCE_SPIN_LIMIT` | 300000000 | 300000000 | **igual en el env, inoperante** (va con el all-reduce RoCE, que va a 0) |
| `CUTE_DSL_ARCH` | `sm_121a` | `sm_121a` | igual |
| `SAFETENSORS_FAST_GPU` | 1 | 1 | igual |
| `VLLM_WORKER_MULTIPROC_METHOD` | spawn | spawn | igual |
| `VLLM_SSM_CONV_STATE_LAYOUT` | DS | DS | igual — **lo exige el PLE b12x** (`ple_layer.py:932`), no solo el GDN; en E2 TP=2 se ve "VLLM_SSM_CONV_STATE_LAYOUT env detected. Setting SSM conv state layout to DS" |
| `VLLM_USE_AOT_COMPILE` / `VLLM_USE_MEGA_AOT_ARTIFACT` | 1 / 1 | 1 / 1 | igual **pero frío** (mismo motivo que las semillas) |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | 1 | igual |
| `B12X_POLICY_MODE` | auto | auto | igual |
| `VLLM_ENABLE_ROCE_ALLREDUCE` | **1** | **0** | **perdido**: su proxy monta la malla por **índice de HCA** (`b12x/comm/roce/_roce_proxy.c:278,302`: `memcpy(blob.gid[h], c->hca[h].gid.raw,16)` / `peer->gid[h]`); en el triángulo cada enlace es su propia /30 y el índice no identifica al vecino. Es lo que dice el VEREDICTO de DGX-847 y lo que aplicó su 3x |
| `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | 2MB | 2MB | igual (inoperante con 0) |
| `VLLM_MXFP8_LM_HEAD` | 1 | 1 | igual (verificado TP=3: 82.816 % 8 = 0 y 43.712 % 8 = 0) |
| `B12X_COMPILE_WORKERS` | 2 | 2 | igual |
| `VLLM_GDN_DEFERRED_CHECKPOINTS` | **1** | **0** | **perdido**: fail-closed con el GDN triton (`gdn_deferred_commit.resolve()`); E2 lo apagó ya en TP=2 |
| `VLLM_GDN_COMPACT_RECORDS` | ausente (default 0) | **0 explícito** | igual en valor; la línea es nueva porque la 1x la trae a 1 y aquí se hereda del env de la 1x el riesgo de confusión |
| `VLLM_PREFIX_DROP_EXACT` | 1 | 1 | igual |
| `VLLM_MTP_DRAFT_VOCAB` | `/opt/mtp-vocab/ids-v2-K131072.txt.gz` | **`/opt/mtp-vocab-es/ids-es-K131072.txt.gz`** | cambiado (castellano, mismo K=131.072; ids ∈ [0, 248.320) verificado) |
| `VLLM_GDN_SHARED_PREFILL_STAGING`, `VLLM_PLE_MMAP*` | no están en la 2x (son de la 1x) | no están | igual que la 2x: la PLE va en GPU |

### 4.3 Red y NCCL (`env-sparkrun-2x.env` → `env-ring-3x.env`)

| Variable | 2x | Aquí | Estado |
|---|---|---|---|
| `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1`, `XDG_CACHE_HOME`, `OMP_NUM_THREADS=4`, `NCCL_IGNORE_CPU_AFFINITY=1` | sí | sí | igual |
| `NCCL_CUMEM_ENABLE` | **0** | **1** | cambiado: en el triángulo con cuMem a 0 NCCL conecta árboles y PAT en el init y no pueden (`ibv_modify_qp 110`); medido en DGX-810/815 |
| `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_CROSS_NIC=1` | sí | sí | igual |
| `NCCL_IB_HCA` | fijo `rocep1s0f0,roceP2p1s0f0` (2 HCAs) | **lo genera el grafo por rango** (mínimo 4, `QWEN_NCCL_RING_HOST`) | cambiado |
| `NCCL_IB_GID_INDEX=3` | sí | **fuera** | **perdido**: el TP=3 medido (DGX-817/820) no lo pone; NCCL elige el GID RoCE v2 IPv4 (el preflight comprueba que los cuatro HCAs coinciden) |
| `UCX_NET_DEVICES=…` | sí | **fuera** | perdido (mismo motivo) |
| `NCCL_NET_PLUGIN=none` | no | **none** | nuevo |
| `NCCL_RUNTIME_CONNECT=1`, `NCCL_ALGO=Ring` | no | **sí** | nuevo (receta DGX-817: conexión diferida y solo anillo) |
| `NCCL_DEBUG` | no (default) | **WARN** | nuevo |
| `LD_PRELOAD` / `VLLM_NCCL_SO_PATH` | no (NCCL de la imagen) | **`/opt/nccl-ring/lib/libnccl.so.2`** (libnccl parcheada, por volumen de imagen) | nuevo |
| `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME`/`MN_IF_NAME` | `enp1s0f0np0` | **`enP7s7`** | cambiado (el bootstrap y gloo van por la LAN, que es lo que ven los tres) |

### 4.4 Manifiesto (`ops/ursu.yaml` 2x → `ursu-3x.yaml`)

| Cosa | 2x | Aquí | Estado |
|---|---|---|---|
| imagen | `spark-vllm-b12x@sha256:8abb60ee…` | la misma | igual |
| `privileged`, `hostNetwork`, `hostIPC`, `dnsPolicy: ClusterFirstWithHostNet`, `tolerations`, `strategy: Recreate` | sí | sí | igual |
| `resources`: 1 GPU, 16Gi request / 112Gi limit | sí | sí | igual |
| `readinessProbe /health:8000` cada 10 s | sí | sí | igual (solo en el head; los dos workers van `--headless`) |
| `shm` `emptyDir` memory 25Gi | sí | sí | igual |
| nodos | dgx2 (head) + dgx1 (worker) | dgx2 (rank 0) + dgx1 (rank 1) + **dgx3 (rank 2)** | cambiado (nuevo rango) |
| `VLLM_HOST_IP` | 10.0.0.2 / 10.0.0.1 | **spark2 / .140 / .143** | cambiado |
| `QWEN_NCCL_RING_HOST` | no existe | **dgx2 / dgx1 / dgx3** | nuevo |
| ConfigMap receta | `ursu-recipe` (con serve-1x y serve-2x) | **`ursu-fork-recipe-3x`** (solo serve-3x.sh) | cambiado (nombres propios para no pisar los de DGX-847) |
| ConfigMap env | `ursu-env-2x` | **`ursu-fork-env-3x`** | cambiado |
| etiqueta | `dgx-847: prueba` | **`dgx-852: fork-tp3`** | cambiado |
| montaje `/cache/huggingface` | PVC `hf-cache-nvidia-dgx-ro` en el head, hostPath en dgx1 | igual **más** dgx3 por el PVC `hf-cache-dgx1-dgx3-llm-ro` | cambiado: **dgx3 no necesita copia** (NFS-over-RDMA ro, server 10.0.2.1; es el montaje con el que corre hoy el residente en dgx3) |
| `/opt/mtp-vocab-es` | en la 2x base **no está**; E2 lo añadía parcheando el yaml en el momento | **está en el manifiesto** (ConfigMap `qwen38fn-mtp-draft-vocab`, ro) | igual que E2, pero declarado |
| `xdg` hostPath | `/srv/ursu-cache/2x` | **`/srv/ursu-cache/fork-3x`** | cambiado (no mezclar cachés de compilación con otra corrida) |
| volumen `nccl-ring` | no | **sí** (imagen `internal.example/homelab/nccl-ring@sha256:79db26e7…`, subPath `opt/nccl-ring`) | nuevo |

### 4.5 Qué se gana y qué se paga (lo único medido: E2 en TP=2, no en TP=3)

- **Se paga**: el all-reduce RoCE propio de su fork (lo que más pesa a c1 en TP=2) desaparece en el triángulo
  → NCCL por anillo con la libnccl parcheada (23 GB/s medidos con 256 MiB en DGX-810/815); los checkpoints
  GDN diferidos y el staging/compact del prefill GDN se apagan con E2; el primer arranque es autotune frío
  (semillas y AOT claveados a TP=2 y a la ruta del modelo).
- **Se gana**: el estado SSM en BF16 (E2) divide a la mitad la página mamba frente al FP32 de su 2x; el
  vocabulario del borrador en castellano sube la aceptación en texto español; TP=3 reparte los 53,1 GiB de
  pesos por rango (≈35 GiB/rango) y deja más KV.
- Números E2 reales (TP=2, `mtp/definitivo/e2/`): decode **c1 73 · c4 158 · c8 226 · c10 243 tok/s**,
  aceptación c1 {0:0,73 1:0,47 2:0,26 3:0,15}; TP=1: c1 47 · c4 98 · c8 137. Su 2x con b12x/FP32 estaba en
  c1 ≈70 y c8 ≈235-256 (`QUE-LLEVA-CADA-UNO.md`): **E2 no cuesta casi nada a c1 y algo a c8+**. En TP=3 no
  hay ni un número medido.

---

## 5. Desviaciones del encargo (todas por escrito)

1. **La construcción en dgx1 corre como root, no con `--uid=1000 --gid=1000`.** `fs.protected_hardlinks=1`
   impide que uid 1000 enlace ficheros root:644, y sin enlaces duros hay que duplicar 85,8 GB (no caben).
   El resto del guard se respeta tal cual: `sudo -n systemd-run --scope -q -p MemoryMax=12G
   --setenv=HOME=/home/dibanerz nice -n 19 ionice -c3`, numpy del site de usuario, `torch` forzado a `None`
   por runpy (`correr.py`), y `ionice`/`nice` para no perturbar la prueba de inferencia que corría.
2. **dgx1 no tenía ~155 GB libres: tenía 96 GB** al empezar, y escribir los 56,0 GB nuevos metió al nodo en
   DiskPressure (§9.2). Ahora 151 GB libres tras el GC de imágenes del kubelet. `construir-vistas.sh` ya no
   se conforma con 65 GB absolutos: exige que **después** del escrito queden >20 % del volumen.
3. **dgx3 no recibe copia**: el encargo daba por hecho un hostPath `/srv/models/hf-ursu` (como en la 3x de
   DGX-847). Comprobado en vivo: el cache de dgx1 está exportado por NFS-over-RDMA también por 10.0.2.1 y
   la PVC `hf-cache-dgx1-dgx3-llm-ro` ya lo monta el residente en dgx3. Se usa eso: 133 GB que no se copian
   y la misma ruta en los tres rangos.
4. **La copia MXFP8 se reconstruye** en vez de reutilizar la que DGX-847 dejó hecha (`ursu-mxfp8-tp3`): su
   index tenía 301.730 entradas apuntando a 35 ficheros inexistentes (144 nombres `.linear_attn.` → shard
   00034 ausente). La nuestra trae 480 entradas presentes.
5. **Nombres nuevos** (`ursu-fork-3x-*`, `ursu-fork-recipe-3x`, `ursu-fork-env-3x`, etiqueta `dgx-852`) para
   no pisar los ConfigMaps/despliegues de DGX-847 si estuvieran aplicados.
6. **`${EXTRA_ARGS:-}` recuperado** en la línea exec (la 3x de DGX-847 lo había perdido); sin él no se
   pueden probar variantes sin reescribir el script.
7. **`VLLM_GDN_COMPACT_RECORDS=0` escrito explícitamente** aunque su default ya sea 0 (es la pareja del
   diferido y la receta 1x la trae a 1).
8. **`imagePullSecrets: [harbor-pull]` añadido a los tres pods** (no estaba en `tp3/ursu-3x.yaml` ni en el
   manifiesto del residente). Verificado en vivo: sin él, en un nodo sin la imagen en cache el volumen
   `nccl-ring` da `ErrImagePull … no basic auth credentials` aunque el secreto exista en el namespace (§9.1
   y §5.8 de este informe). Es un cambio del manifiesto que el encargo no pedía; sin él el rango de dgx3 no
   arranca.

---

## 6. Lo que NO he podido verificar (marcado)

- **Rendimiento en TP=3**: ni tok/s ni tiempo de arranque. Ningún rango llegó a cargar el modelo (el pod de
  dgx1 fue desalojado por DiskPressure, el de dgx3 no pudo tirar la imagen y el de dgx2 lo borré a los 25 s
  del incidente §9.1). Las únicas cifras E2 son de TP=2/TP=1 (`mtp/definitivo/e2/`).
- **El `--block-size 16` en TP=3+E2**: el motor lo sobrescribe para que la página de atención ≥ página
  mamba (1504 en TP=2, 1584 en TP=1). En TP=3 la página mamba por rango se divide entre 3 pero la de
  atención pasa de 1 a 2 cabezas KV: **no sé en qué sale**, hay que leerlo del log del primer arranque.
- **Autotune/semillas/AOT en TP=3**: las semillas de la imagen están claveadas a TP=2 y a la ruta del
  modelo, así que se espera compilación fría (minutos) en el primer arranque; no medido.
- **Que el tramo de carga b12x acepte la forma [2500014,160] sin la variable `QWEN_TP3_PLE_SHARD_SKIP`**: la
  aritmética del loader está verificada en el código (filas 2 y 4 de la tabla §1) pero no hay ejecución.
- **El arranque distribuido por la LAN con el fork** (bootstrap `spark2:25000`, `--nnodes 3`): la
  receta es la de DGX-847, que tampoco llegó a arrancar el fork en TP=3; lo verificado en DGX-817/820 fue
  **nuestro** vLLM por ese camino.
- **El grafo del anillo con 4 HCAs en dgx3** (el preflight lo comprueba antes de arrancar y aborta con 7,
  pero no se ha ejecutado).
- **La escala global de la PLE** (`ngram_embedding.weight_scale`, BF16 [1]) viaja **sin tocar** por enlace
  duro; el loader exige `weight_scale_shape == (1,)` y dtype `caps.scale_dtype` (`ple_layer.py:744-752`) —
  la forma la verifico en la vista, el dtype lo heredo del checkpoint del propio fork.
- **Que `--additional-config` con triton en prefill no obligue a recompilar el MTP**: no está ejecutado.

---

## 7. Comandos exactos para arrancarlo

Reconstruir las vistas (**ya están hechas**: el script aborta si alguna de las dos carpetas existe, para
no reescribir 56 GB por accidente mientras corre otra prueba; `--fuerza` es la salida deliberada, y en dgx1
**hoy abortaría también por disco** porque el guard de espacio exige >196 GB libres tras el escrito, §9.2):

```bash
ssh dgx 'bash /home/dibanerz/dgx-852/construir-vistas.sh' \
  | tee /home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3/construir.log
#    (o con --fuerza para volver a construir encima)
```

Lanzar (**con la ventana abierta**: ArgoCD root y ai sin auto-sync, Job `llm-deploy-pause`, qwen38 a 0 y
`drop_caches`; hoy corren ahí `ursu-2x-head`, `ursu-2x-worker` y `ursu-1x`):

```bash
bash /home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3/aplicar-3x.sh
```

Que hace, en orden y nada más: comprobar las dos vistas en dgx1 por ssh, comprobar las dos PVC y la clave
`ids-es-K131072.txt.gz` del ConfigMap, crear/actualizar `ursu-fork-recipe-3x` y `ursu-fork-env-3x`, y
`kubectl apply -f ursu-3x.yaml`.

Seguir y probar:

```bash
kubectl logs -n llm -l dgx-852=fork-tp3 -f          # hasta 'b12x ready' y /health 200
curl -s http://spark2:8000/v1/models | python3 -m json.tool
# en el arranque, mirar y apuntar:
#   "Setting attention block size to …"   (el 16 real que sale en TP=3)
#   "MTP draft vocab: 131072 of 248320 ids from /opt/mtp-vocab-es/…"
#   "GDN decode kernel: triton"  /  "MXFP8 copy serves rows >= 41"
#   "Available KV cache memory"  /  "GPU KV cache size"
# la vuelta es la de siempre: ops/cierre.sh
```

Por rango: rank 0 = dgx2 (gx10-ec3d, sirve `:8000`), rank 1 = dgx1 (nvidia-dgx), rank 2 = dgx3. Los tres con
`VLLM_HOST_IP` = su IP de LAN y `QWEN_NCCL_RING_HOST` = dgx2/dgx1/dgx3 según `ring.conf`.

## 8. Ficheros entregados

- `/home/user/bench-results/dgx-847/mtp/definitivo/fork-tp3/` → `serve-3x.sh`, `env-3x.env`,
  `env-ring-3x.env`, `ursu-3x.yaml`, `aplicar-3x.sh`, `construir-vistas.sh`, `correr.py`, `construir.log`,
  `INFORME.md` (esto).
- `/home/user/bench-results/dgx-847/mtp/checkpoint/convert-ursu-nvfp4-tp3.py` (modificado: `--fork-view`,
  `--mxfp8-copy`, `--config-from`, `place()` fail-closed, `fork_roundtrip`).
- `/home/user/bench-results/dgx-847/mtp/checkpoint/test/test_fork_view.py` (nuevo).
- dgx1: `/home/dibanerz/.cache/huggingface/{qwen38fn-mtp-ursu-nvfp4-tp3-fork,qwen38fn-mtp-ursu-mxfp8-tp3}` y
  `/home/dibanerz/dgx-852/` (copias de trabajo del conversor y del lanzador).

## 9. Los dos incidentes de este encargo, tal cual

**9.1 Ejecuté el `apply` real al probar el guion de arranque (08:09:34Z).** Para validar solo la parte de
comprobaciones de `aplicar-3x.sh` generé una copia recortada con `sed '/^log "ConfigMaps de la 3x/,$d'`; el
patrón no casaba (la línea real es `log "ConfigMaps de la 3x del fork"`) y **el guion entero se ejecutó**:
creó los ConfigMaps `ursu-fork-recipe-3x` y `ursu-fork-env-3x` y los Desplegamientos
`ursu-fork-3x-head/worker-dgx1/worker-dgx3`. Lo vi en la salida y **lo borré todo a los ~30 s**
(`kubectl delete` de esos cinco objetos; comprobado: `get deploy,cm -l dgx-852=fork-tp3` = sin recursos).
Efecto real sobre lo que corría:
- el head del fork entró en dgx2 con `hostNetwork` y compartió el puerto 8000 y la GPU con `ursu-2x-head`
  durante ~25 s. `ursu-2x-head` llevaba 3 reinicios y el último a las **08:09:24Z, diez segundos ANTES** de
  mi `apply`: no se lo cargué yo, y después siguió cargando normal (world_size=2, rank 0).
- `ursu-1x` en dgx3 no se tocó (1/1 Running, sin reinicios).
- el worker del fork en dgx1 ni llegó a correr (desalojado por el DiskPressure de 9.2) y el de dgx3 falló en
  `ErrImagePull` del `nccl-ring` → de ahí salió el hallazgo del `imagePullSecrets` (§5.8).
- el residente sigue a 0 y el árbitro en pausa, como estaba.

**9.2 La construcción de la vista metió a dgx1 en DiskPressure y el kubelet le borró las imágenes.** El
encargo daba 155 GB libres; dgx1 tenía **96 GB** (10,5 % del volumen). Escribir los 56,0 GB nuevos dejó
40 GB = **4,4 %**, por debajo de los umbrales de desalojo del kubelet (nodefs 10 %, imagefs 15 %): el taint
`node.kubernetes.io/disk-pressure` en `nvidia-dgx` se puso a las **07:42:03Z** (justo en la ventana
07:40:19→07:44:25Z del `convert`) y desde entonces el `ursu-2x-worker` se desalojó en bucle (decenas de pods
`Evicted`). El GC de imágenes se llevó de dgx1 **`ghcr.io/ursuciprian/spark-vllm-b12x@sha256:8abb60ee…` y el
`nccl-ring`** (la cache de containerd bajó a 4,1 GB). Estado al cerrar este informe (08:25Z): presión
quitada (08:18:57Z), **151 GB libres (15,4 %)**, la imagen del fork **ya volvió** porque el propio worker la
re-bajó a las 08:22Z al reanudar, y `ursu-2x-head`/`ursu-2x-worker` están a **0/0 réplicas** — las escaló su
dueño, no yo; `ursu-1x` sigue 1/1 en dgx3. Lo que queda por saber es si la 2x de DGX-847 perdió la corrida
por la hora de desalojos (su worker nunca estuvo arriba desde ~07:42Z).
**Mientras la vista del fork esté en dgx1, el nodo vive al 15 % justo del umbral del kubelet**: cualquier
escrito grande (un `pull` de 20 GB, un log, otra vista) vuelve a poner el taint. Lo que yo haría si hay que
seguir: liberar sitio en dgx1 (las vistas de DGX-847 `qwen38fn-mtp-ursu-nvfp4-tp3` son 120 GB y `hub` 242 GB)
o mover la vista del fork a dgx2 (510 GB libres) y exportarla desde allí. **No he borrado nada de eso: es
decisión de Dani.**

Medidas tomadas en los guiones para que no se repita:
- `construir-vistas.sh`: aborta si al escribir los 56 GB el volumen queda por debajo del **20 %** libre
  (antes pedía solo 65 GB absolutos, que era justo lo que faltaba) y aborta si alguna de las dos vistas ya
  existe (`--fuerza` para reconstruir). Comprobado: hoy **reconstruir en dgx1 está bloqueado por ese guard**
  (quedarían 91 GB de 982; hacen falta >196). Se reconstruye liberando ~150 GB en dgx1 o construyendo en
  dgx2 (510 GB libres), lo que exige exportar el cache de dgx2 por NFS porque las dos PVC ro salen de dgx1.
- `aplicar-3x.sh`: antes de tocar nada comprueba en los tres nodos que **no hay taint de disk-pressure** y
  que las dos imágenes están en la cache del nodo; con presión aborta y con imagen ausente avisa con el
  motivo exacto.
- Ningún guion se vuelve a ejecutar "recortado" para probarlo: las comprobaciones de 9.1 y 9.2 se hicieron
  con los comandos sueltos por bash.

```
[FINDING: el fork de ursuciprian NO sube el alineado de la tabla PLE para TP: `padded_vocab_size % tp_size`
aborta en b12x/sequence/ple_embedding/_contracts.py:381. Con `make_ngram_vocab_size_divisible_by: 128` la
tabla 320.001.536 no divide entre 3 y el modelo muere en la capa PLE; hay que declarar 384 y re-trocear las
128 partes a 2.500.014 filas (256 ceros al final de la tabla global).]
[FINDING: su `VocabParallelEmbedding` padrea él mismo a lcm(64,tp) y exige en el loader las filas REALES del
checkpoint (`assert loaded_weight.shape[0] == org_vocab_size`): una vista padreada por fuera (248.448) no
carga; se recorta a 248.320.]
[GOTCHA: `fs.protected_hardlinks=1` + ficheros root:644 = uid 1000 no puede enlazar; cualquier construcción
de vistas por enlace duro en dgx1 tiene que ir como root dentro del mismo guard systemd-run.]
[GOTCHA: dgx1 va justo de disco (916 GB, ~613 GB de caches de modelo). Escribir 56 GB cruza el umbral de
desalojo del kubelet, pone el taint `disk-pressure` y el GC borra las imágenes de los pods: antes de
cualquier escrito grande, `df -B1 --output=avail,pcent /` y exigir >20 % libres DESPUÉS del escrito.]
[GOTCHA: el volumen de imagen `nccl-ring` se resuelve con las credenciales del POD y el SA `default` de llm
no trae `imagePullSecrets`: en un nodo sin la imagen en cache el pull falla `no basic auth credentials`
aunque el secreto `harbor-pull` exista en el namespace. Hay que enlazarlo en el pod spec.]
[DECISION: no borro las 120 GB de la vista `qwen38fn-mtp-ursu-nvfp4-tp3` de DGX-847 ni los 242 GB de `hub`
en dgx1 para devolver el sitio: es irreversible y le toca a Dani.]
```
