# DGX-847 · ¿puede el fork de ursuciprian correr TP=3 en los tres Sparks?

**Veredicto: viable con cambios. El cambio no es su fork, es el checkpoint.**

- Con el checkpoint tal cual, TP=3 no arranca: cinco formas no dividen por 3 y lo hacen abortar (cabezas KV 2,
  cabezas clave GDN 16, intermedio del experto compartido 640, intermedio de los expertos dirigidos 640 y el
  `padded_vocab_size` de la PLE). Ninguna es capricho de su código: son `assert`/`divide()` de vLLM.
- Con una **vista TP=3 del checkpoint** (el mismo trabajo que hicimos para nuestro FP8 en DGX-780, ampliado a sus
  formatos NVFP4/MXFP8 y al pad del MoE) **no hace falta ni un parche de código en su fork**: ya resuelve solo el
  padreo del vocab, el recorte de logits, el `fc` del MTP y el draft vocab de 131.072, y el resto lo trae la vista.
- Su allreduce RoCE one-shot **no sirve en el triángulo a 3 rangos**: monta la malla por índice de HCA
  (`_roce_proxy.c:302` toma `peer->gid[h]`, con `ROCE_MAX_HCAS 2` y `roce_oneshot.py:241` quedándose con las dos
  primeras HCAs) y nuestros tres enlaces son dgx1P0–dgx2P0, dgx1P1–dgx3P0 y dgx2P1–dgx3P1. Se apaga con
  `VLLM_ENABLE_ROCE_ALLREDUCE=0` y cae limpio a PYNCCL; ahí entra nuestra libnccl del anillo (imagen `nccl-ring`,
  grafo por rango, `NCCL_ALGO=Ring`), que ya usamos en TP=3 medido.
- El arranque bootstrap cambia de red: `10.0.0.2` (master del 2x) es la /30 dgx1↔dgx2 y dgx3 no la ve; se pasa a
  la LAN (`spark2`, `enP7s7`), como el residente.

**Esfuerzo.** Una jornada de sesión para el conversor (los dos formatos de escala, el pad 640→768, las filas de la
PLE, `vocab_size` sin tocar) + la corrida de ~115 GB de copia de bytes y publicarla en dgx1 y dgx3 + la ventana.
El manifiesto, la receta y los env están escritos y validados contra el API server: `ursu-3x.yaml`,
`recipe/serve-3x.sh`, `recipe/env-3x.env`, `recipe/env-ring-3x.env`, `aplicar-3x.sh`. Sin la vista publicada,
`serve-3x.sh` aborta con `exit 6` y un log que dice exactamente qué falta. Detalle en `CONVERTIR.md`.

**Velocidad esperada.** c1 **75–95 tok/s, lo más probable ~85** (estimación) frente a los 70,4–72,5 medidos de su
TP=2; c8/c10 **230–260**, sin ganancia clara porque las dos stacks ya están en ese techo. Base de la estimación:
el paso TP=2→TP=3 en nuestro stack valió **+33 % en c1** (34,1–38,4 en DGX-781 → 46,9–49,1 en DGX-817/820) y
**nada por encima de c8** (187 → 207–239); la ventaja de ellos en c1 (70 con TP=2) no viene del TP sino de sus
kernels b12x, MTP=4 y el one-shot RoCE, y al perder RoCE se paga la latencia de NCCL por el anillo
(~96 colectivos por token, 5 KB cada uno: +1,5–3 ms/token, estimación). Riesgo: si esa latencia pesa más de lo
previsto, c1 queda en 60–70, o sea igual que su TP=2 y TP=3 no compensa. Arranque: autotune **frío** (la semilla
de la imagen está claveada a `tensor_parallel: 2` y a la ruta del modelo, así que fallan todos sus registros) —
10–25 min estimados, por medir.

Nota de alcance: TP=3 ocupa los tres Sparks, así que no libera dgx3; la comparación que busca DGX-847
(adoptar su receta y poder parar dgx3) se decide igual con el 2x + 1x que ya está medido.

## 1 · Qué no divide por 3 y cómo lo resolvimos nosotros (evidencia)

Todo con el `config.json` real de su snapshot `16c9bd54` (oculto 2560, `head_dim` 256, 24 Q / 2 KV, GDN 16:48 de
128, 512 expertos con intermedio 640, compartido 640, vocab 248.320, PLE `ngram_vocab_size_base` 20.000.000,
`heads_per_ngram` 8, `split_ngram_parts` 128, `make_ngram_vocab_size_divisible_by` 128).

| forma | valor | qué falla en su fork | cómo se arregló en el nuestro |
|---|---|---|---|
| cabezas KV | 2 | `linear.py:1082` `divide(tp_size, num_kv_heads)` y `qsa.py:745` «TP size must divide replicated QSA KV heads» | vista: 2→6 `[k0,k0,k0,k1,k1,k1]` (2 por rango) |
| cabezas clave GDN | 16 | `qwen_gdn_linear_attn.py:871-872` `divide(num_k_heads, tp)` (y el Caps del kernel b12x de decode); valor 48 sí dividía | vista: 16:48→24:72 con cabezas a cero, grupo 1:3 viaja junto |
| experto compartido | 640 | `linear.py:738` `assert all(output_size % tp == 0)` en MergedColumnParallel | vista: 640→768 con ceros |
| expertos dirigidos | 640 | `fused_moe/config.py:1349` `assert intermediate_size % tp_size == 0`; y **no se puede huir por EP**: `fused_moe/b12x.py:568-572` rechaza EP y `oracle/nvfp4.py:255-305` aborta con «backend 'b12x' does not support the deployment configuration» si se lo pides explícito | el nuestro va por EP (171/171/170, parche `QWEN_FI_EP_UNEVEN`); aquí toca pad 640→768 y quedarse en b12x |
| PLE | `padded_vocab_size % 3` | `b12x/sequence/ple_embedding/_contracts.py:375-380` raise; `padded = align_up(sum(primes), divisor)` se calcula, no se lee (`ple_hash/geometry.py:95`), y el loader pide `ceil(padded/128)` filas por parte y compara exacto (`ple_layer.py:826-849`) | vista: alineamiento 128→384 (320.001.536→320.001.792) y **re-particionar las 128 partes cruzando ficheros**: cada parte pasa de 2.500.012 a 2.500.014 filas, sus fronteras se desplazan 2 filas por parte y la parte i de la vista nace de las partes i e i+1 del checkpoint. No basta «añadir 2 filas al final de cada parte». Nuestro `QWEN_TP3_PLE_SHARD_SKIP` no vale aquí (exige cobertura completa `ple_layer.py:624-635`) |
| vocab / lm_head | 248.320 | **nada**: `vocab_parallel_embedding.py:336` `math.lcm(padding_size, tp_size)` → 248.448 y `logits_processor.py:361/389/441` recorta y enmascara el relleno | nuestro parche `QWEN_TP3_VOCAB_MASK` es innecesario en su fork; la vista deja `vocab_size` 248.320 y padrea solo los tensores |
| MTP `fc` | 3×2560 | **nada**: `mtp.py:223/232` ya es ReplicatedLinear | nuestro `QWEN_TP3_MTP_FC_REPLICATED` es innecesario |
| draft vocab | 131.072 | **nada**: `mtp.py:684` `DraftVocabLMHead(ParallelLMHead)` padrea a 131.136 → 43.712 por rango, con `scatter_draft_logits` (`:792-840`) | sin parche |
| `hc_lowrank` 320, `hc_count` 4 | no dividen | **nada**: `hyperconnection.py:336-390` ReplicatedLinear / `disable_tp=True` | sin parche |
| indexer 4/1 cabezas | no divide | **nada**: ReplicatedLinear (DGX-780) | sin parche |
| vision | 16 cabezas / 4304 | pide `--mm-encoder-tp-mode data`, que la receta 2x ya lleva | sin cambio |

Comprobado sobre el checkpoint ya bajado en el x86 (97 GiB, rev 16c9bd54, 301.838 tensores):

- Las familias están **mezcladas capa a capa dentro del mismo fichero**: `self_attn.{q,k,v,o}`, `in_proj_a/b` y el
  experto compartido salen **MXFP8** (peso F8_E4M3 + escala U8 cada 32); `in_proj_qkv/z/out_proj`, los expertos
  dirigidos y la tabla PLE salen **NVFP4** (peso U8 empaquetado 2 por byte + escala F8_E4M3 cada 16 + un
  `weight_scale_2` F32 escalar que no se parte); el MTP entero y `embed_tokens`/`lm_head` salen **BF16**. La
  familia se deduce del dtype, no del nombre del tensor.
- La rama PLE no la parte el TP: `key_proj`/`value_proj` son `ReplicatedLinear` (`ple_layer.py:971-986`) y
  `conv1d`/`norm_conv`/`norm_key`/`norm_query` son buffers del grupo (`:989-992`); lo único que se corta es la
  tabla ngram. Igual el indexer: `index_qk_proj` es `ReplicatedLinear` (`nvidia/indexer_qsa.py:45`).
- La copia MXFP8 del segundo checkpoint es solo el shard 35 (480 tensores: GDN y self_attn de las 48 capas). El
  `lm_head` MXFP8 no está en ningún shard: es cuantización online del bf16 del principal (`VLLM_MXFP8_LM_HEAD=1`).

## 2 · Alineamientos de los kernels b12x y cachés

No hay ningún `assert` de múltiplo de 64/128 que el corte por 3 de la vista viole; los que existen son de 32 y de
256, y todos los cocientes caen justos: MoE por rango 256 (`fused_moe/b12x.py:500-520`, intermedio %32 y oculto
%256), escalas NVFP4 cada 16 (768/16 = 48 → 16 por rango) y MXFP8 `[1,32]` (82.816 %32 = 0, 3.072 %32 = 0),
GDN 8/24 cabezas por rango con `head_dim` 128 intacto. `_tp_shard` de la copia MXFP8
(`kernels/linear/nvfp4/b12x.py:95-106`) es la única que corta a mano: exige `n % tp == 0` en cada `output_sizes`
por eso el segundo checkpoint tiene que estar padreado igual que el primero.

La caché de planes **sí está claveada a TP=2**: los 7 ficheros de `/opt/b12x-seed/preparation` llevan en
`identity.namespace` `tensor_parallel: 2` junto a `model`/`draft_model` (ruta resuelta del snapshot), `dtype`,
`kv_cache_dtype`, `pipeline_parallel`, `speculative_method`, `decode_context_parallel`. En TP=3 fallan los 289–682
registros de cada fichero por los dos ejes (TP y ruta nueva del modelo), así que el arranque es autotune frío. No
es un bloqueo, es tiempo. El formato `--load-format b12x` no asume 2: es un plugin registrado
(`b12x/integration/vllm/loader.py:290-292`) que enruta shards por nombre desde el index.

## 3 · Apagar el RoCE y qué más asumía 2 nodos

- `cuda_communicator.py:76`: `use_roce_allreduce = use_custom_allreduce and envs.VLLM_ENABLE_ROCE_ALLREDUCE`; con
  0 no se construye el comunicador y PYNCCL ya existe (`pynccl_comm` se crea con `world_size > 1`). En el log del
  2x se ve el orden real: `Using ['B12X_ROCENANTE','PYNCCL'] … group 'tp:0'` → al quitar el primero queda NCCL sin
  tocar nada más. El límite de 2 MB (`VLLM_ROCE_ALLREDUCE_MAX_SIZE`) ya mandaba lo grande a NCCL.
- El one-shot acepta 2..16 rangos (`roce_oneshot.py:43`, guarda en `:217`), así que no es un tope de mundo: es la **malla por
  índice de HCA** (`post_op` reparte cada peer entre los dispositivos por índice, QP[h][peer]
  con el GID del peer en ese mismo índice). Nuestro triángulo no la da; `B12X_ROCE_HCA` a un solo HCA tampoco
  salva a dgx3.
- Lo que sí asumía 2: los transports `tp2_remote_push` del allreduce PCIe (`b12x/comm/pcie/_oneshot_cute.py:195`
  exige `world_size == 2`; `pcie_allreduce.py:78` idem) — no nos afectan, son de un solo nodo y el propio
  RoCE ya se salta los grupos de un nodo (`b12x_roce_all_reduce.py`, «group is single-node»).
- Banderas de la receta que asumían 2 nodos y cambian: `--nnodes 2` → 3, `--master-addr 10.0.0.2` → LAN,
  `VLLM_HOST_IP` 10.0.0.x → lan-nodex, `NCCL_SOCKET_IFNAME/GLOO/TP = enp1s0f0np0` → `enP7s7`,
  `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` + `NCCL_IB_GID_INDEX=3` + `UCX_NET_DEVICES` fuera (los pone el generador
  del grafo por rango), `NCCL_CUMEM_ENABLE=0` → 1 con `NCCL_RUNTIME_CONNECT=1` y `NCCL_ALGO=Ring`.
  Todo lo demás (`--moe-backend b12x`, MTP=4, `--max-num-seqs 16`, 262.144, fp8 KV, `B12X_AUTOTUNE=1`,
  `VLLM_MXFP8_LM_HEAD=1`, draft vocab) queda igual.

## 4 · Pendiente de la ventana (no se puede verificar sin GPU)

- Que la vista cargue entera en su fork (`verify --tp 3` dentro de la imagen) y las formas reales por rango.
- Tiempo del autotune frío y si `VLLM_USE_AOT_COMPILE`/`VLLM_USE_MEGA_AOT_ARTIFACT` también fallan con TP=3.
- Coste real del allreduce NCCL por el anillo con 96 colectivos pequeños por token (la estimación de arriba).
- Numéricos y calidad con el relleno (cabezas a cero, 128 filas de vocab a `-inf`, MoE con 128 columnas de ceros):
  las 7 suites de calidad son la prueba.
- De qué fichero sale la copia MXFP8 del `lm_head` con `VLLM_MXFP8_LM_HEAD=1`.
- Que `--gpu-memory-utilization 0.8` y 262.144 de contexto quepan con la tabla PLE en GPU a un tercio (a TP=2
  cargaba 53,1 GiB por rango en 60 s).

## Ficheros que dejo

```
~/bench-results/dgx-847/tp3/
  VEREDICTO.md        esto
  CONVERTIR.md        la vista TP=3 del checkpoint: qué cambiar en el conversor, formas y tamaños
  convert-tp3-base.py base sobre la que trabajar (nuestro convert.py de DGX-780, 1102 líneas)
  ursu-3x.yaml        3 Deployments (dgx2 head + dgx1 + dgx3), anillo nccl-ring por volumen de imagen
  recipe/serve-3x.sh  la receta 2x con TP=3, LAN, grafo por rango y preflight
  recipe/env-3x.env   env de la receta con RoCE apagado y las rutas de la vista
  recipe/env-ring-3x.env  env de arranque (bootstrap LAN + receta NCCL de DGX-817)
  aplicar-3x.sh       crea los ConfigMaps y aplica; comprueba antes que la vista esté publicada
```
