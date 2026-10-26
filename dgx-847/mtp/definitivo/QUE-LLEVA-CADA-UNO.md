# DGX-852 · qué lleva cada configuración (14-10-2026, sacado de ficheros y logs)

Fuentes: receta de ursu `ops/recipe/{serve,env}-{1x,2x}`, código de su fork (`fork-src/`, vLLM 20260926.dev28+g884b4ff68),
nuestra imagen (`mtp/checkpoint/imagen-residente/`, vLLM 0.28.1rc1.dev628+g2a02f6efe), `launch-ursu-b12x.sh` y los
"non-default args" de cada arranque.

| | Producción hoy | Ursu TP=2 (su fork) | Nuestro TP=3 A1 + vocab | Nuestro TP=2 / TP=1 «su sistema» |
|---|---|---|---|---|
| vLLM | el nuestro | su fork | el nuestro | el nuestro |
| Checkpoint | NVIDIA FP8 híbrido | ursu NVFP4 + copia MXFP8 (shard 35 de local-inference-lab) para lotes ≥41 tokens | ursu NVFP4, vista TP=3 (rellena) | ursu NVFP4, vista tp12 (sin relleno) |
| PLE | FP8, disco (mmap) | NVFP4 en GPU (en 1 Spark: NVFP4 en disco) | FP8, disco | FP8, disco (TP=1: 47,7 GiB) |
| Estado SSM / bloque de caché | BF16 / 832 | FP32 / 2864 | BF16 / 832 | BF16 / **1680** |
| GDN (prefill y decode) | nuestro (triton + cuda) | b12x (exige FP32) | nuestro | nuestro |
| GEMM | FlashInfer | b12x | b12x | b12x |
| MoE | EP, FlashInfer CUTLASS | b12x sin EP, su b12x (más nuevo que 1.3.0) | EP, FlashInfer CUTLASS | b12x sin EP, 1.3.0 + su micro.py |
| LM head / hiperconexiones | BF16 | MXFP8 / MXFP8 | BF16 | BF16 |
| All-reduce | NCCL (anillo propio) | **RoCE propio** (≤2 MB) | NCCL | NCCL (TP=2 por el puerto 0) |
| MTP | k=1, sin vocabulario | k=4, vocab ids-v2 | k=2, vocab castellano | k=4 (w31), vocab castellano |
| Planificador | prioridad, 10 seqs, 4096 tokens/paso | decode-aware, 4 prefills paralelos, 16 seqs, 8192 tokens/paso | prioridad, 10, 4096 | prioridad, 10, 4096 |
| Caché de prefijo extra | — | drop-exact, checkpoints GDN diferidos | — | — |
| Arranque rápido (AOT + semillas) | no | sí (5-9 min) | no | no |
| Parser de tools / razonamiento | qwen3_coder / low | qwen3_xml / medium | qwen3_coder | qwen3_coder |
| KV | 48 GiB fijo | util 0.8 | 48 GiB = 3,55 M tokens | TP=1: 19,1 GiB = 1,28 M tokens (perfilado, util 0.88) |
| Decode c1 / c4 / c8 (tok/s) | 49 / — / — (k=1) | 70 / — / 235-256 | 60 / 158 / 269 | midiendo |

## Lo que su sistema tiene y nuestra pila NO (no portado)
1. All-reduce por RoCE propio (`VLLM_ENABLE_ROCE_ALLREDUCE`): lo que más pesa en TP=2 a c1.
2. Su b12x nuevo (política `auto`, plan de rutas en host): llevamos 1.3.0 + solo su `micro.py`.
3. Copia MXFP8 de los pesos para lotes ≥41 tokens (`VLLM_B12X_NVFP4_MXFP8_CHECKPOINT`).
4. LM head y hiperconexiones en MXFP8.
5. GDN b12x entero con estado BF16: el **prefill** b12x exige FP32 (`b12x/sequence/gdn_prefill/_impl.py:63-64`) y la prueba E de DGX-847 ni arrancó. El **decode** b12x sí acepta BF16 (`gdn_decode/_impl.py:102-106`): DGX-852 lo hace correr con prefill triton relajando solo la política de selección → `gdn-mix/` (variantas A0/A/B, parche, plan de GPU).
6. Planificador decode-aware, prefills paralelos, 16 seqs y 8192 tokens/paso.
7. Caché de prefijo: drop-exact, checkpoints GDN diferidos, staging/compact del prefill GDN.
8. AOT + semillas de compilación (nuestro vLLM tiene las llaves; falta activarlas, va en la PR).
9. PLE en NVFP4 (la nuestra en FP8 ocupa el doble).

## Lo que nuestra pila tiene y su fork NO
- Estado SSM en BF16. El bloque de 832 es solo de TP=3 (la vista replica las cabezas KV); en TP=2 y TP=1 es 1680.
- TP=3 (parches de EP desigual, MTP FC replicado, máscara de vocabulario, PLE por shards).
- Vocabulario del borrador en castellano, planificador por prioridad (las prioridades de LiteLLM), umbral de prefill largo.

## Su fork con nuestro estado BF16 (prueba E2 de DGX-847, 13-10)
Su receta con estado BF16 y el GDN triton en vez del b12x: bloque 1504, primera respuesta −25 % a c1 y −15 % a c8,
generación +6-27 %, prefill de 2K −16 %. Funciona en TP=2 y TP=1; no existe en TP=3.

## Decode b12x con prefill triton sobre estado BF16 (DGX-852, 14-10 — pendiente de GPU)
`mtp/definitivo/gdn-mix/`: `INFORME.md` (analisis con file:line), `PLAN-GPU.md` (lo ejecuta la duena de los Sparks),
`patch_gdn_mix.py` + `serve-gdn-mix.sh` (parche anclado fail-closed por `QWEN_GDN_MIX=A|B`) y `greedy_compare.py`.
A = decode b12x BF16 + prefill triton sin diferidos: viable con una sola relajacion de politica. B = mas
diferidos/compactos: aborta a c2+ por el guard de registros compactos (`qwen_gdn_linear_attn.py:3010-3014`); la B
servible es B2 (prefill b12x en BF16), que es trabajo de kernel en `b12x`, no un parche. A0 (decode `cuda`
fusionado + prefill triton) no lleva parche: lo permite la politica de vLLM y solo falta que el op este compilado.
