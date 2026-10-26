# DGX-852 · b12x MoE nvfp4 — fix LISTO

Config A (e768 + `--moe-backend b12x` + `--linear-backend b12x`, sin EP, MTP k=2) revienta con
`CUDA error: an illegal memory access` durante `capture_model`. Aquí está la causa acotada, el
parche de arranque, el micro-repro para un Spark y lo que queda por confirmar en GPU.

## Causa (acotada con evidencia)

**El kernel `direct-micro` nvfp4 de b12x 1.3.0, en su primera ejecución con enrutado real
(M=6), lanza un acceso ilegal.** No es el relleno 640→768, ni el all-reduce, ni el dense b12x,
ni el dynamic grouped-GEMM.

La cadena de evidencia:

1. **El tamaño que falla es M=6, no M=56.** La barra de progreso de `a-arranque.log` (16:11:53 →
   16:12:18) muestra `Capturing CUDA graphs (PIECEWISE): ... 9/14` y luego el ERROR. Los descs van
   ordenados de mayor a menor `[56,48,40,32,30,24,16,12,8,6,4,3,2,1]`
   (`v1/worker/gpu/cudagraph_utils.py`, `capture()`): pasaron 9 tamaños (56..8) y el 10º es **6**.
   El traceback (`cudagraph_utils.py:386 forward_fn(CUDAGraphMode.NONE)` → `model.py:328` →
   `qwen_gdn_linear_attn.py:932` out_proj → `linear.py:1769` → NCCL) es el paso EAGER previo a la
   captura; el IMA es asíncrono y aparece en el primer sync del forward, no en el kernel culpable.
2. **M=6 es la frontera de la banda micro.** `select_tp_moe_backend` elige micro si
   `num_tokens ≤ 8 y num_tokens*topk < cutover`, con cutover por defecto **64 pares**
   (`b12x-1.3.0/b12x/moe/fused_moe/_impl.py:1853` y `:1856`). Con topk=10: M=6 → 60 pares →
   **micro**; M=8 → 80 → **dynamic**. La forma del problema soporta micro (k=2560 → 5 segmentos
   K ≤ 12, `micro.py:85-101`), así que la banda está activa.
3. **El micro nunca se ejercitó con enrutado real antes de M=6.** El warmup b12x lanzó **1**
   firma MoE (`a-arranque.log:659 "Warmed up 1 b12x MoE kernel signature(s)"`) y
   `warmup_launches` de la glue usa `topk_ids = arange(topk).expand(tokens,-1)` → solo los
   expertos 0..9, uniformes (`imagen-residente/vllm/model_executor/layers/fused_moe/b12x.py`,
   `warmup_launches`). El enrutador real sobre M=6 activa hasta **60 expertos distintos con 1-6
   filas cada uno**. La profiling de memoria estaba saltada (`kv_cache_memory_bytes`,
   `a-arranque.log:641`), no hubo forward real previo.
4. **El dynamic con enrutado real YA funcionaba en este arranque**: las 9 capturas 56..8
   (dynamic tile16, `a-arranque.log`) pasaron con el router real. El compilado del dynamic es
   independiente de `m` (la clave de cache de `_get_dynamic_kernel` excluye `m`/`max_rows`,
   `_impl.py:9411-9433`) → mandar la banda micro a dynamic reutiliza un kernel probado.
5. **La geometría del plan/launch es autoconsistente con nuestros números** (E=512, k=2560,
   n=2·256=512, topk=10, capturas 1..56, prefill 4096): escalera de tiles
   (`_impl.py:_select_dynamic_tile_mn`, umbrales 15/48/96·E), task geometry
   (`_impl.py:2159`), arena (`_plan_core_workspace`, `:3185-3250`), barrier
   (`barrier_slots=routed_rows`, `:3015`), y `direct_routing` está muerto para nosotros
   (`_DIRECT_ROUTING_MAX_ROUTED_ROWS=32`, `_impl.py:1860`; routed_rows=60..560 > 32). Codificado
   como test en CPU: `test-geometria.py` (ejecutado, OK). El OOB no viene de tamaños mal
   calculados por la glue; está dentro del runtime del kernel micro de 1.3.0.
6. **El fork reescribió exactamente esta banda y su receta no pisa el kernel de 1.3.0**:
   route-plan en host con Triton (`_dynamic_route_plan_kernel`, fork `_impl.py:1946+`), kernels
   materializados `nvfp4_phase1/phase2`, `planned_tile_m` para forzar acuerdo plan/launch, y el
   knob `nvfp4_materialize_intermediate` (`fork _tuning.py:518`) que `B12X_POLICY_MODE=auto`
   activa en decode (`ops/recipe/env-2x.env`). Nada de eso existe en la rueda 1.3.0 del
   wheelhouse. Que upstream reconstruyera el path es la mejor señal de que el fused micro/dynamic
   de 1.3.0 en decode era lo roto.

**Por qué NO es el padding e768**: el rank que revienta es TP0 (`Worker_TP0 pid=962`), y la
columna rellenada (640..767) cae en el rango del rank 2 (512..767); el rank 0 no tiene filas
de padding (`VEREDICTO.md §9`). Y A1 (misma imagen, MoE FLASHINFER_CUTLASS) arranca y sirve.

## Parche

`bloque-wa.sh` — bloque al estilo de `QWEN38_B12X` de `launch-nvfp4.sh`, para insertar justo
antes del `exec taskset -c "$CPUS" vllm serve "${ARGS[@]}"`. Es 100% env + validación
fail-closed (rc=73) contra la rueda/vllm instalados; no toca el grafo ni reinicia nada.

- `QWEN38_B12X_MOE_WA=cutover0` (recomendado): `B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=0` →
  `select_tp_moe_backend` pide `routed_rows < 0`, la banda micro desaparece y **todo** va al
  dynamic tile16 ya validado con enrutado real en el arranque A. Comprueba en CPU (sin CUDA)
  que la rueda instalada elige dynamic para M=1..8 y aborta con 73 si no. Coste: decode M≤8 más
  lento (micro era ~2.2× según `_resolve_workspace_layout`); para medir b12x MoE da igual.
- `QWEN38_B12X_MOE_WA=a16` (alternativa): `VLLM_B12X_MOE_FP4_FORCE_A16=1` → el oracle nvfp4
  (`fused_moe/oracle/nvfp4.py:176-179`) pasa el MoE a la familia w4a16 (Trellis), ni micro ni
  dynamic nvfp4. **Solo con `QWEN38_B12X_MTP31=1`** (ver abajo), porque usa la misma entrada
  `("nvfp4", None)` de la glue.
- `QWEN38_B12X_MTP31=1` (bug latente aparte): la glue nuestra declara
  `("nvfp4", None): ("w4a16","modelopt_nvfp4","w13")` (`imagen-residente/.../fused_moe/b12x.py:45`)
  y el fork `"w31"` (`fork-src/.../fused_moe/b12x.py:51`). En b12x `"w31"` = gate primero
  (`micro.py:450-451`) y vLLM guarda w13 = [gate; up] → la nuestra etiqueta mal la capa MoE del
  MTP (grupo w4a16). El arranque A no llegó a pisarla (revienta antes, en la captura del target);
  con `cutover0` el MTP sí llegará a correr y, sin este fix, los borradores salen con gate/up
  intercambiados (aceptación ~0). El bloque reescribe la línea (ancla exacta, idempotente,
  fail-closed si la ancla no aparece una sola vez). Validar con P5 del repro antes de fiarse.

Probado en CPU: `bash -n` del launch-nvfp4.sh con el bloque insertado; rc=73 con valor
desconocido y sin `QWEN38_B12X`; ancla w13→w31 aplicada e idempotente sobre copia de la glue;
`test-geometria.py` en verde (bandas, tamaños de arena y cotas de tiles para todos los tamaños
de captura).

## Activación

1. Insertar `bloque-wa.sh` en `launch-nvfp4.sh` antes del `exec` (una vez; el bloque es
   idempotente y no toca configs).
2. En el Deployment del residente: `QWEN38_B12X_MOE_WA: cutover0` (y `QWEN38_B12X_MTP31: "1"`
   cuando P5 confirme el layout). Rollback: quitar las llaves.
3. Nada más cambia: misma vista e768, mismos args, misma rueda.

## Micro-repro (un Spark, minutos)

- `micro-repro.py`: construye `Qwen4ExpSparseMoeBlock` de UNA capa con los pesos NVFP4 reales de
  la vista e768 vía el loader de vLLM (AutoWeightsLoader + ModelOptMixed + B12xExperts +
  process_weights_after_loading), con la vista de rank 0 de TP=3 inyectada parcheando
  `get_tensor_model_parallel_world_size/rank` (layer.py:50, config.py:1124) — n=512, I_pp=256,
  E=512, topk=10. Forward eager en los 14 tamaños de captura con `CUDA_LAUNCH_BLOCKING=1`.
  `--hidden zeros` replica el warmup; `--hidden randn` el enrutado real; `--ref` dequantiza
  nvfp4 en fp32 y da el error relativo (detecta el swap w13/w31 en `--layer mtp`); salva
  `/tmp/repro-*.pt` para diff contra flashinfer_cutlass.
- `job-micro-repro.yaml`: Job en dgx3 (toleración `dedicated=llm:NoSchedule`, imagen del
  residente por sha, PVC `hf-cache-dgx1-dgx3-llm-ro` en `/cache/huggingface` readOnly, hostPath
  `/tmp` para cache JIT y salidas, `nvidia.com/gpu: 1`), instala la rueda con el mismo comando
  que `QWEN38_B12X` y corre P1..P5. **No aplicado** (Sparks ocupados); crear antes el ConfigMap
  `dgx852-micro-repro` con el script (comando en la cabecera del YAML).

## Lo que no se ha podido comprobar sin GPU

- Que P2 reproduzca el crash exactamente en M=6 (la atribución a micro es por descarte
  estructural + frontera de banda, no por traza del kernel culpable).
- Que `cutover0` elimine el IMA en el arranque real (M=1..6 sobre dynamic es el mismo compilado
  que pasó 56..8, pero no se ha ejecutado).
- Que la familia w4a16 (`a16`) arranque con este checkpoint (repack Trellis de 512×768).
- Que `"w31"` sea el layout correcto para la capa MTP w4a16 (P5 lo dirime contra referencia).

## Si P2 NO reproduce, trazas concretas que hacen falta

1. Arranque A con `QWEN38_B12X_MOE_WA=''` + `CUDA_LAUNCH_BLOCKING=1` (planificado): el launch
   culpable sale en la propia traza — con eso se sabe si es `tp_moe_compact_micro_launch`,
   `tp_moe_dynamic_launch` u otro, y en qué M.
2. Sobre el repro, el pase que reproduzca bajo `compute-sanitizer --tool memcheck
   --print-access-stack python3 micro-repro.py --hidden randn --sizes 6` (la imagen lleva el
   CUDA toolkit): da la dirección y el access stack exactos del OOB.
3. `B12X_DYNAMIC_WORK_SOURCE=grid` en un arranque adicional si el culpable resultara ser el
   dynamic (descarta el protocolo `materialized_queue` de publicación diferida).
