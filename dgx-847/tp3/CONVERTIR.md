# DGX-847 · la vista TP=3 del checkpoint de ursuciprian (lo que falta para poder arrancar)

Su fork no necesita ningún parche de código en TP=3 si el checkpoint llega con las formas ya dividibles por 3.
Esa vista es el trabajo pendiente, y es el mismo que hicimos para nuestro FP8 (DGX-780): la base está en
`convert-tp3-base.py` (copia de `convert.py` del ConfigMap `llm/qwen38fn-tp3`, 1102 líneas, copia de bytes sin
torch). Aquí se describe qué hay que cambiarle; abajo está cada forma con su número por rango.

## Qué hay que extender de `convert-tp3-base.py`

1. **Formatos de peso.** La base solo conoce FP8 por bloques 128×128 (`BLOCK = 128`). El checkpoint de ursu trae
   tres familias (config.json → `quantization_config.config_groups`):
   - `group_nvfp4_routed_experts` y `group_w4a16_nvfp4_*`: NVFP4, `group_size 16` → `weight` empaquetado 2 por
     byte (`[filas, cols/2]` uint8), `weight_scale` 1 byte por cada 16 de columna (`[filas, cols/16]` fp8e4m3) y
     `weight_scale_2` escalar por tensor. Al mover filas o columnas enteras viajan juntas: `weight`, sus
     `weight_scale` y el `weight_scale_2` (no se toca).
   - `group_mxfp8_shared_experts` / `group_mxfp8_attention`: MXFP8, `weight_block_size [1,32]` → `weight`
     fp8e4m3 y `weight_scale` uint8 con una columna de escala cada 32.
   - el resto bf16/fp8 sin bloques.
   Están **mezcladas dentro del mismo fichero y del mismo tipo de capa** (`self_attn` sale MXFP8 y
   `linear_attn.in_proj_qkv` NVFP4 en el mismo shard 00035), así que en el conversor la familia sale del `dtype`
   del tensor, no del nombre: peso U8 = NVFP4 (2 valores por byte), peso F8_E4M3 = MXFP8, peso BF16 = sin
   cuantizar; escala F8_E4M3 = NVFP4 (cada 16), escala U8 = MXFP8 (cada 32).
   Con relleno a 768 y 248448 todos los cocientes por rango caen a múltiplos de 32 (tabla abajo), así que las
   dos familias entran enteras en cada rango; no hay que recuantizar nada.
2. **El pad que faltaba: `moe_intermediate_size` 640 → 768.** La base no lo hacía porque nuestro TP=3 parte los
   expertos por EP (171/171/170) y nunca parte el intermedio. El `--moe-backend b12x` de ursu **rechaza EP**
   (`fused_moe/b12x.py:568-572`: `not use_ep and ep_size == 1`), así que aquí sí se parte: gate/up por filas y
   down por columnas, en las 48 capas y en el MTP (`mtp.layers.0.mlp.experts`, W4A16).
3. **`vocab_size` NO se cambia en el config.** La base escribe 248448 en `vocab_size` y arma el parche
   `QWEN_TP3_VOCAB_MASK`. En su fork eso es un error: `logits_processor.py:361` recorta a `org_vocab_size` y
   `:389/:441` enmascaran con `-inf` `num_org_vocab_padding`. Dejando `vocab_size` 248320 con tensores de
   248448 filas, su fork padrea solo (`vocab_parallel_embedding.py:336`, `math.lcm(64, 3)` → 248448) y las 128
   filas de relleno quedan a `-inf` sin parche. Se mantienen los tensores padreados (por el corte por rangos de
   la copia MXFP8) y el config con el valor real.
4. **PLE.** `b12x/sequence/ple_embedding/_contracts.py:377-386` exige `padded_vocab_size % tp_size == 0`, con
   `padded_vocab_size = align_up(sum(prime_sizes), make_ngram_vocab_size_divisible_by)`. Las `prime_sizes` vienen
   del propio checkpoint (`ngram_heads_vocab_sizes`), así que lo único que tocamos es el alineamiento:
   - si `sum(prime_sizes) % 3 == 0`, se deja `128` y **no se toca la tabla**;
   - si no, sube a `384` en el config de la vista y entonces `ceil(padded/128)` crece (en el de ursu, de
     2.500.012 a 2.500.014 filas por parte: 320.001.536 → 320.001.792, 256 filas de relleno al final de la
     tabla). **No es «añadir 2 filas al final de cada parte»**: las partes son bloques uniformes de la tabla
     global (`checkpoint_start = shard_index * shard_size`, `ple_layer.py:826-831`), así que al cambiar
     `shard_size` cambian todas las fronteras y la parte i de la vista se construye con las filas globales
     `[i·2500014, (i+1)·2500014)`, que salen de las partes i **e i+1** del checkpoint (los 128 ficheros están
     repartidos en 16 shards: es lectura cruzada de fichero). Verificado byte a byte sobre 100 filas de las
     partes 0,1,16,…,113 con peso y escala.
     El parche `QWEN_TP3_PLE_SHARD_SKIP` de nuestro launch.sh **no es válido aquí**: `ple_layer.py:624-635`
     exige cobertura completa de las filas del rango y su receta sirve la tabla en GPU, no desde NVMe.
5. **El segundo checkpoint, solo una pieza.** `VLLM_B12X_NVFP4_MXFP8_CHECKPOINT` se usa por
   `kernels/linear/nvfp4/b12x.py:110-155`, que corta por rango y compara con `output_size_per_partition`
   (`_tp_shard:95-106`: en ColumnParallel exige `n % tp == 0` para cada `output_sizes`, y en RowParallel que las
   columnas por rango sean múltiplo de 32). La pieza que baja la receta 2x es `model-00035-of-00036.safetensors`
   (2,7 GB, 480 claves = 240 tensores: `in_proj_a/b/qkv/z` y `out_proj` de las 36 capas GDN y
   `q/k/v/o_proj` + `indexer.index_qk_proj` de las 12 de atención, cada uno con su escala) más el index: esa
   pieza y el index son lo único que hay que convertir, y se convierte **con las mismas unidades** que el
   principal (filas 512→1536 en KV, 10.240→15.360 en qkv, columnas 6.144→9.216 en out_proj), porque el loader
   corta la copia contra la partición de la capa NVFP4 y compara `output_size_per_partition`.
   El `lm_head` MXFP8 no viene de aquí: `lm_head.weight` está en el shard 00034 en BF16 y **sin** `weight_scale`
   (verificado en el index) — `VLLM_MXFP8_LM_HEAD=1` lo cuantiza online sobre el bf16 del checkpoint principal.
6. **Ficheros que se copian tal cual**: todo lo que no sea `config.json` ni `*.safetensors` (tokenizer,
   `chat_template.jinja`, `generation_config.json`, `hf_quant_config.json`, `export-manifest.json`). El
   `config.json` de la vista cambia solo `num_key_value_heads`, `linear_num_key_heads`, `linear_num_value_heads`,
   `shared_expert_intermediate_size`, `moe_intermediate_size` y, si hace falta,
   `make_ngram_vocab_size_divisible_by`.
7. **`verify`.** La base ya tiene `verify --tp 3` (equivalencia en tensores reales + carga en las capas reales de
   la imagen con tp=3). Hay que correrlo dentro de `ghcr.io/ursuciprian/spark-vllm-b12x` (sin GPU basta la parte
   de formas; la carga de capas pide un Spark, o sea la ventana).

## Formas y cocientes por rango (TP=3, oculto 2560, head_dim 256, GDN 128)

| qué | checkpoint | vista TP=3 | por rango | regla que se cumple |
|---|---|---|---|---|
| cabezas KV (QSA y full) | 2 | 6 `[k0,k0,k0,k1,k1,k1]` | 2 | `linear.py:1079-1085` `divide(tp, kv)` y `qsa.py:735-746` |
| GDN cabezas clave | 16 | 24 (8 a cero) | 8 | `qwen_gdn_linear_attn.py:871` `divide(num_k_heads, tp)` |
| GDN cabezas valor | 48 | 72 | 24 | `:753/:758` `divide(num_v_heads, tp)`; 48 ya dividía, se sube con las clave para que el grupo 1:3 viaje junto |
| `in_proj_qkv` / `z` / `out_proj` | 10.240 / 6.144 / 6.144 | 15.360 / 9.216 / 9.216 | 5.120 / 3.072 / 3.072 | múltiplos de 32 para las escalas `[1,32]` de la copia MXFP8 |
| `shared_expert_intermediate_size` | 640 | 768 | 256 | `linear.py:738` `output_size % tp == 0` |
| `moe_intermediate_size` (512 expertos + MTP) | 640 | 768 | 256 | `fused_moe/config.py:1349` y `fused_moe/b12x.py:500-520` (por rango %32) |
| `embed_tokens` / `lm_head` | 248.320 filas | 248.448 filas (config sigue 248.320) | 82.816 | `vocab_parallel_embedding.py:336` `lcm(64,3)=192`; 82.816 %32 = 0 |
| draft vocab MTP | 131.072 | sin cambio | 43.712 | `mtp.py:684` `DraftVocabLMHead` padrea a 131.136 solo |
| tabla PLE | `align_up(total,128)` | `align_up(total,384)` si `total %3 != 0` | `padded/3` | `_contracts.py:377-386`; partes de `ceil(padded/128)` filas |
| `hidden_size` 2560 | sin partir | igual | 2560 | en RowParallel es salida y en ColumnParallel es entrada: nunca se divide |
| `hc_lowrank` 320, `hc_count` 4 | sin partir | igual | 320 | `hyperconnection.py:336-390` ReplicatedLinear / `disable_tp=True` |
| indexer 4 cabezas / 1 KV | sin partir | igual | 4 | ReplicatedLinear (DGX-780) |
| MTP `fc` | sin partir | igual | — | `mtp.py:223/232` ya es ReplicatedLinear |

## Dónde y cuánto

- Origen: dgx1 `/home/dibanerz/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE`
  (107 GB) y la pieza 00035 del `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (2,7 GB).
- Salida: ~115 GB (los pads suman ~20 % del MoE, que es la mayor parte de los bytes).
- Publicar en `dgx1:/home/dibanerz/.cache/huggingface/tp3/` (el PVC `hf-cache-nvidia-dgx-ro` se la sirve a dgx2)
  y en `dgx3:/srv/models/hf-ursu/tp3/`, con los nombres que esperan `recipe/env-3x.env`
  (`ursu-qwen38fn-tp3`, `ursu-mxfp8-tp3`). Hueco de sobra: 390 GB libres en dgx1, 425 en dgx3, 552 en dgx2 para
  construir. Es trabajo de CPU y I/O, sin GPU: se puede hacer mientras la batería mide, pero sin tocar los
  Sparks si están cargados (la copia por NFS compite con la lectura de pesos).
- Esfuerzo: el conversor ya existe y hace copia de bytes con unidades de filas/columnas; lo nuevo son los dos
  formatos de escala, el pad del intermedio de MoE y las filas de la PLE. Una sesión con `convert-tp3-base.py`
  delante lo saca en un día, más la corrida (unas horas de I/O) y el `verify` de formas.
