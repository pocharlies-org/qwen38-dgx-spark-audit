# INFORME DGX-852 — GDN mixto en la imagen de ursuciprian: estado BF16 con decode `b12x` (prefill `triton` en A/B, prefill `b12x` en B2)

Sesión de análisis sobre el código extraído (solo lectura) de su imagen:
`/home/user/bench-results/dgx-847/fork-src/usr/local/lib/python3.12/dist-packages/{vllm,b12x}`.
Todas las rutas de abajo son relativas a ese `dist-packages`. Nada de este
informe toca el clúster: lo que necesita GPU está marcado en §7 y el plan es
`PLAN-GPU.md`.

Referencias de partida: E2 = `triton/triton` + estado SSM BF16 + diferidos y
compactos a 0 (arrancado con `ops/prueba-e2.sh`, arriba ahora mismo en
`spark2:8000` el 2x y `spark3:8000` el 1x). Receta de ursu =
`b12x/b12x` + estado FP32 (`ops/recipe/serve-2x.sh:28`, `env-2x.env:19`).

---

## 0. Resumen

| variante | qué es | veredicto de código |
|---|---|---|
| **A0** | prefill `triton` + decode `cuda` (el `fused_gdn_decode_post_conv_mtp` de vLLM), **sin parche** | la política de vLLM **ya lo permite** (`qwen_gdn_linear_attn.py:328-344`: el raise solo mira `"b12x"`). Solo falta que el op esté compilado en su imagen, y **eso se comprueba solo**: pedido explícitamente por `--gdn-decode-kernel cuda`, si no es soportable **aborta** con `ValueError` (`:813-817`) en vez de caer a triton. **Medir primero.** |
| **A** | prefill `triton` + decode `b12x` con estado BF16, diferidos y compactos **apagados** | **viable con UNA relajación de política** (el selector de backend). Todo lo demás ya lo soporta el código: el decode `b12x` acepta BF16 por sí solo (`b12x/sequence/gdn_decode/_impl.py:102-106`), el estado es el mismo tensor con las mismas strides (`_cute_kernels.py:1472-1477`), la conv la hace vLLM igual que en E2 (`:2774`+`causal_conv1d_update`), y los lotes mixtos caen al camino triton que E2 ya sirvió. Parcheado, testeado en CPU, fail-closed. |
| **B** | A + `VLLM_GDN_COMPACT_RECORDS=1` + `VLLM_GDN_DEFERRED_CHECKPOINTS=1` | **no es servible por el camino B1** (prefill triton). El propio fork lo aborta en cuanto un lote mixto tiene filas especulativas (`qwen_gdn_linear_attn.py:3010-3014`), y la razón es real: con diferidos la base del bloque en curso está **incompleta** entre fronteras de bloque y el camino triton no sabe reproducir registros. B solo da trabajo a c1. La versión servible es **B2** (prefill `b12x` + decode `b12x` con estado BF16), y **sí cabe como parche de arranque**: FP32 no está en el algoritmo del kernel, está en dos guardas, en el tipo del puntero de estado (que el prefill fijaba a `Float32` a mano mientras el decode ya lo derivaba del binding) y en la conversión del store. Está hecho: modo `B2` del parcheador, 21 ediciones de prefill sobre 6 ficheros (24 en total con las 3 de decode), probadas en CPU. **Sin verificar en GPU.** |
| **B2** | prefill `b12x` + decode `b12x` + estado BF16 + diferidos y compactos a 1 (la receta de ursu con el estado en BF16) | **servible por diseño**: es el camino `b12x/b12x` para el que `b12x_mixed` y `refuse_reasons` ya existen, así que no necesita levantar ni la selección ni los motivos de rechazo, solo que el kernel de prefill acepte la base BF16 (§5.1). Lotes mixtos a c≥2 por la ruta diseñada, no por el camino triton que aborta. |

Recomendación: **A0 → A → B2** (con `test_b2_gpu.py` antes de servirlo). El
detalle de por qué B1 no, con la cadena de evidencia, en §4; es el hallazgo del
ticket. Corrección de este informe: la primera versión afirmaba que B2 no era
alcanzable con un parche de arranque; no lo comprobé contra el kernel y estaba
mal (§4.3).

---

## 1. Qué es exactamente "el mix" y quién decide

El selector único es `_resolve_gdn_backend_selection` (`:299-344`). Lee
`additional_config["gdn_prefill_backend"]` y `additional_config["gdn_decode_kernel"]`
(este último también puede venir del flag CLI `--gdn-decode-kernel`, que
`arg_utils.py:2676-2677` vuelca dentro de `additional_config`; por eso da igual
pasarlo de una forma o de otra). Con `prefill="triton"`, `decode="b12x"` llega a
`:328` y **salta el `ValueError` de `:330-334`**:

```python
if prefill == "b12x" or decode == "b12x":
    if prefill not in ("auto", "b12x") or decode not in (None, "b12x"):
        raise ValueError("b12x GDN prefill and decode must be selected together; ...")
    return "b12x", "b12x", explicitly_configured
```

Ese raise es **el único bloqueo de A**. No es una comprobación técnica: unas
líneas más arriba (`:320-325`) el propio código ya admite combinaciones mixtas
(`flashinfer` + `triton`, cualquier prefill + `cuda`). La política es "o los dos
b12x, o ninguno", y está escrita para el caso en que **ambos** son b12x porque
solo entonces existe la ruta mixta (`_forward_core_b12x_mixed`, `:3028`) y el
perfil de CUDA graphs `ALWAYS`.

Un matiz que importa para A0 y para el auto-default: si no se pide decode
explícitamente, `:344` devuelve `decode or "cuda"`, o sea **cuda por defecto**, y
entonces `explicitly_configured=False`. Con `--gdn-decode-kernel cuda` explícito
el fallo es ruidoso; con decode sin pedir el fallo es silencioso (log + triton,
`:813-820`). De ahí que el plan exija la línea `GDN decode kernel: ...` (`:851`)
en el log de arranque para saber lo que se está midiendo.

---

## 2. A0 — el decode CUDA fusionado, sin tocar nada

* **Política**: permitida (arriba). `("triton", "cuda")` sale por `:344`.
* **Requisitos del op**: `_fused_gdn_decode_unsupported_reason` (`:1409-1432`)
  pide modelo BF16, `conv_state` BF16, estado en `FUSED_GDN_STATE_DTYPES =
  (float32, bfloat16)` (`:218`), K=V=128, `num_v_heads % num_k_heads` en
  (1,2,3,4,8), computabilidad ≥ 8.0 y
  `hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")` (`:1430`). El env
  `-2x` de la receta ya cumple los de dtype (modelo `--dtype bfloat16`, estado
  BF16 con `--mamba-ssm-cache-dtype bfloat16`).
* **Qué hace en cada lote**: la ruta fusionada exige `num_prefills == 0`
  (`:2996-3009`); en lote mixto cae al mismo fallthrough triton que A y que E2
  (`:3015-3026`). O sea A0 tiene **exactamente la misma conducta en lotes mixtos
  que A**, pero con un kernel de decode CUDA de vLLM en los pasos uniformes.
* **¿Está compilado en su imagen?** No lo puedo saber desde la extracción: en
  `fork-src` **no hay ni un `.so`**; el `RECORD` del wheel declara
  `vllm/_C_stable_libtorch.abi3.so` (165 MB) pero ese fichero no vino. El
  contenedor lo comprueba en un segundo (§1 de `PLAN-GPU.md`):
  `python3 -I -c "import torch, vllm._C; print(hasattr(torch.ops._C,'fused_gdn_decode_post_conv_mtp'))"`.
* **Vale la pena medirlo primero**: cero parche, cero riesgo de estado (es
  código de vLLM que ya está en la ruta soportada), y si su op CUDA rinde como
  el b12x en decode, A deja de merecer la pena.

---

## 3. Variante A — todo lo que toca el mix, con evidencia

### 3.1 Selección y construcción del layer

* `:328-335` es el bloqueo → lo relaja la edición 1 del parche (modo A y B),
  condicionada a `QWEN_GDN_MIX in (A,B)` **en runtime**, no en el texto del
  fichero: sin la variable de entorno puesta, el código parcheado vuelve a
  comportarse como el original (eso lo comprueba `test_selection_logic.py`).
* `:822` `enable_fused_gdn_decode = decode in ("b12x","cuda")` → en A es True.
* `:833-842` `gdn_deferred_commit.resolve(...)` / `resolve_compact(...)`: con
  `DEFERRED=0` y `COMPACT=0` devuelven False **sin preguntar nada**
  (`gdn_deferred_commit.py:161-167`, `176-178`: `if not requested(): return False`).
  Por eso A no necesita relajar ni el prefill ni el dtype en esa capa. Ojo:
  `env-2x.env:19` trae `VLLM_GDN_DEFERRED_CHECKPOINTS=1` de casa, y `resolve()`
  es fail-closed con `prefill != b12x` (`:120-124`) → **A tiene que ponerlo a 0
  explícitamente**, que es lo que hace `serve-gdn-mix.sh`.
* `:843` `_initialize_b12x_gdn_decode` (`:858-887`): `state_index_columns =
  num_spec + 1` (5 con k=4) y, **solo si compact**, reserva el ancho de
  registros. En A no hay registros.

### 3.2 Builder de metadatos (`vllm/v1/attention/backends/gdn_attn.py`)

* `self._b12x_mixed` **se construye solo si `gdn_prefill_backend == "b12x"`**
  (`:203-212`). Con prefill triton es `None` → la rama `_forward_core_b12x_mixed`
  del dispatch (`:2973-2989`) es inalcanzable. No hay nada que parchear ahí: no
  es que la ruta mixta funcione mal, es que **no existe** y el código cae al
  camino triton, que es el que E2 ya usa en producción desde siempre.
* La metadata de chunks de FLA (`prepare_chunk_indices`) se construye
  **precisamente cuando el prefill NO es b12x** (`:648`) → en A se construye,
  como en E2.
* `_b12x_prefill_live_counts` es `None` con prefill triton (`:193-196`), y el
  `VLLM_GDN_UNIFORM_DECODE_META_SKIP=1` del env `-2x` solo afecta al staging de
  `_b12x_mixed` (`:910-928`) → **inofensivo en A** (no hay worklists que saltar).
* Estado de filas: `num_decodes` y `num_spec_decodes` son excluyentes, y cuando
  hay especulativos los decodificados simples **se reclasifican como prefills**
  (`:541-546`), con el comentario "el kernel de prefill maneja secuencias de 1
  token con estado inicial correctamente, produciendo resultados idénticos". Es
  decir: en un lote mixto, A procesa **todo** (prefills y decodes) por triton.

### 3.3 CUDA graphs

`get_cudagraph_support` (`:159-171`) devuelve `ALWAYS` **solo** si
`prefill == decode == "b12x"`; con el mix devuelve `UNIFORM_BATCH`
(`_cudagraph_support`, `:150`). Consecuencia: solo se capturan los lotes
uniformes (todo-decode), y los mixtos van eager — **el mismo perfil que tiene
E2 hoy**, así que A no introduce riesgo nuevo de captura, pero tampoco el
beneficio de `ALWAYS`. Hay que confirmarlo en el log (PLAN §6).

### 3.4 El estado: escribe triton exactamente lo que lee b12x

Esta es la pregunta de fondo, y la respuesta es que **no hay dos estados**:

* Ambos caminos usan el mismo tensor: `recurrent_state = self.kv_cache[1]`
  (`:1338` en `_bind_b12x_gdn_decode`), y el triton escribe
  `ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)`
  (`:2453`), leyendo el inicial de la misma forma (`:2435`).
* El formato lo fija FLA: el estado recurrente es `[N, H, V, K]` con **K
  contiguo** (`vllm/third_party/flash_linear_attention/ops/chunk.py:169,181` y
  `chunk_delta_h.py:114`), o sea por cabeza, fila = dimensión de valor, columna
  = dimensión de clave.
* El kernel b12x **lo exige y lo comprueba al bindear**:
  `_binding_key` (`b12x/sequence/gdn_decode/_cute_kernels.py:1472-1477`)
  comprueba `recurrent_state.stride()[1:] == (V_DIM*K_DIM, K_DIM, 1)` y lanza
  `"recurrent state must be contiguous within each slot"` si no. Y antes, que
  K=V=128 (`:1469-1471`). Es decir: si el layout no casase, **A aborta en el
  primer bind, no corrompe en silencio**.
* El dtype: el kernel sube el estado a FP32 en registros antes de operar
  (`:509-514`) y lo devuelve escribiendo con `self.state_type` (`:700-712`), así
  que BF16 es un dtype de almacenamiento soportado; `Caps` lo admite
  explícitamente **mientras no se pidan checkpoints diferidos**
  (`b12x/sequence/gdn_decode/_impl.py:102-106` frente a `:127-135`).
* La **conv** no es asunto de b12x: `_forward_core_decode_b12x_fused_norm`
  (`:2774`) llama a `causal_conv1d_update` de vLLM con
  `is_conv_state_dim_first()` (que es `VLLM_SSM_CONV_STATE_LAYOUT=DS`) y luego
  pasa `mixed_qkv` ya convolucionado a `_run_b12x_gdn_decode_post_conv`. En el
  `_impl.py` de b12x **no aparece la palabra `conv`**. Por tanto el layout DS
  queda 100 % en manos del código de vLLM, idéntico en E2 y en A.
* Los **índices de estado** que usa el decode b12x para la conv son la columna 0
  de la tabla de bloques (`state_indices[:num_requests, 0]`, `:2792`), la misma
  que usa triton → mismo bloque en curso.
* Las **copias de estado** entre bloques (migración de frontera en modo align y
  export a la caché de prefijo) las hace `postprocess_mamba_fused_kernel` con
  `state_elem_sizes` por estado (`vllm/v1/worker/mamba_utils.py:1586-1589`):
  copias por tamaño de elemento, ajenas al kernel. E2 ya las hace con BF16.
* La geometría del pool **no cambia**: `num_speculative_blocks` (`mamba/abstract.py:93`)
  pasa a 0 **solo con compact**; con compact=0 A tiene la misma tabla de bloques
  que E2 (1 + 4 especulativas). A cambia quién toca el pool, no el pool.

### 3.5 MTP k=4 y verificación especulativa

* El decode `b12x` se usa en dos formas de lote uniforme (`:2899-2917`):
  (a) solo especulativos (`num_decodes == 0 and num_spec_decodes > 0`, con
  `spec_state_indices_tensor` de hasta `state_index_columns = num_spec+1`
  columnas) y (b) solo decodes simples. El replay de los tokens aceptados entra
  por `num_accepted_tokens` (`:1341-1343`, staging en `_initialize_b12x_gdn_decode`).
* En lote mixto con especulativos, A **no** usa b12x: usa el camino triton con
  `spec_sequence_masks` (`:2154` `_forward_core`), que es el que E2 ejecuta hoy
  con k=4 a c8. No hay código nuevo.
* El riesgo propio de A aquí no es de corrección sino de **consistencia
  numérica**: el mismo request puede pasar un step por b12x (lote uniforme) y el
  siguiente por triton (lote mixto). Los dos leen y escriben el mismo estado con
  el mismo layout, así que no se "pierden" tokens; lo que cambia es el redondeo,
  y eso es exactamente lo que mide el greedy contra E2 y el suelo de ruido.

### 3.6 Caché de prefijos

* Modo `align` + `--enable-prefix-caching` (receta `:13-14`), con
  `VLLM_PREFIX_DROP_EXACT=1` (`env-2x.env:20`), que afecta a los checkpoints de
  frontera de request en el coordinador híbrido
  (`vllm/v1/core/kv_cache_coordinator.py:723-736`, `kv_cache_manager.py:235-242`)
  — código que no mira qué kernel GDN se eligió.
* Un golpe de caché lo consume el **prefill** (`initial_state =
  ssm_state[prefill_state_indices]`, `:2435`), que en A es triton igual que en
  E2. La parte nueva es que lo que escribió la vez anterior pudo ser b12x →
  mismo tensor, mismo layout (§3.4), y por eso el caso `cache-cold` vs
  `cache-warm` del comparador es la prueba directa, sin baseline.

### 3.7 Lo que **no** hace falta tocar para A

No es una lista de "no mirado": son los sitios que verifiqué y ya valen.
`_make_b12x_gdn_caps` (`:911-935`) pide `state_dtype` del propio layer (BF16) y
`deferred_checkpoints=False` → Caps legal. `gdn_deferred_commit` no se activa.
`_b12x_records_kwargs` (`:1354-1355`) no pasa registros. `resolve_compact`
devuelve False. El builder no crea `b12x_mixed`. Los kernels de commit
(`gather_gdn_commit_windows_kernel`, `boundary_export_decision_kernel`) no se
llaman porque `deferred.active` es False (`mamba_utils.py:1606-1609`).

---

## 4. Variante B — por qué elegí documentar B1 como roto y B2 como el trabajo real

Dani pidió elegir entre B1 (prefill triton que nunca lee un bloque con registros
en vez de estado) y B2 (que su kernel de prefill acepte BF16), con evidencia.

### 4.1 La premisa de B1 es medio cierta, y la mitad que falta es la que mata

* **Lo que sí es cierto**: con `COMPACT=1` los registros por token **no viven en
  el pool**. Viven en un side buffer FP32 por capa indexado por slot de request
  (`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:206-208` `record_columns =
  num_speculative_tokens if compact_requested() else 0`) y el pool **suelta sus
  bloques especulativos** (`mamba/abstract.py:93` → `num_speculative_blocks=0`).
  La prueba de que sin compact los registros sí estarían dentro del estado:
  `_cute_kernels.py:1494-1500`, donde para `deferred` **no** externo exige
  `recurrent_state.stride(0) >= record_floats`, o sea "los registros por token
  se guardan dentro de un slot de estado". Con `external_records` esa exigencia
  desaparece y lo que se exige es un buffer FP32 `[rows, width]` fila-contiguo
  (`:1478-1493`). **Aquí es donde el guard de FP32 deja de ser necesario**: los
  deltas son FP32 en su buffer; la base puede ser BF16. Por eso el parche levanta
  los tres guardas **solo con `external_records`** (ediciones 4, 5, 6).
* **Lo que falta**: con diferidos, la base del bloque en curso está
  **incompleta** entre fronteras. El commit que la materializa se decide en el
  postprocess de cada step y solo para las peticiones con frontera/export
  (`gdn_deferred_commit.py:309-347` `boundary_export_decision_kernel`, y su
  llamada en `mamba_utils.py:1606-1620`: "Decide, commit, then copy"). Un lote
  mixto que caiga al triton leería esa base vieja, escribiría un estado completo
  a partir de ella y **perdería los registros pendientes en silencio**.

### 4.2 El fork ya lo sabe y aborta

`_forward_core_fused_norm` (`:2954-3026`) en A/B tiene esta forma:

1. `b12x_mixed != None` → ruta mixta b12x (`:2973-2989`). **Inalcanzable con
   prefill triton**: la metadata no se construye (`gdn_attn.py:203-212`) y
   `_forward_core_b12x_mixed` (`:3028-3138`) exige el runner de prefill b12x y
   `gdn_decode_kernel == "b12x"` (`:3042-3043`).
2. `_can_use_b12x_gdn_decode` → **`num_prefills != 0` es un no** (`:2899-2904`).
3. Ruta fusionada CUDA → también exige `num_prefills == 0` (`:2996-3009`).
4. `if self._b12x_gdn_compact_records and attn_metadata.num_spec_decodes:`
   `raise RuntimeError("compact GDN records: spec-decode rows reached a
   non-b12x path")` (`:3010-3014`).
5. else → triton + `_rms_norm_gated_cuda` (`:3015-3026`).

Con MTP k=4 encendido (receta `:27`), cualquier lote con alguien prefillando y
otros decodificando tiene `num_prefills>0` **y** `num_spec_decodes>0` (y si
además hay decodes simples, se reclasifican como prefills, `gdn_attn.py:541-546`)
→ **paso 4 → RuntimeError**. No hay knob en este fork que impida lotes mixtos
(busqué: no existe `mixed_batch`/`no_mixed` en `config/scheduler.py` ni en
`envs.py`; `prefill_policy`/`max_parallel_prefills` controlan el troceo, no la
mezcla).

**Conclusión B1**: levanta, sirve a **c1** (todos los lotes uniformes, nunca
paso 4) y revienta a **c≥2**. El abort es ruidoso y es el resultado del
experimento, no un bug del parche. No lo hago "funcionar" silenciando el guard:
eso convertiría un crash en corrupción silenciosa, que es justo lo que el guard
evita. Queda medido en GPU como prueba negativa deliberada (PLAN §4, paso 8 de
`prueba-gdn-mix.sh`).

### 4.3 B2 es la única B servible — y sí cabe como parche de arranque

Para tener diferidos+compactos con concurrencia hace falta que **el prefill sea
b12x** (así existe `b12x_mixed` y los lotes mixtos van por la ruta diseñada). Y
para que el estado sea BF16 hace falta que su kernel de prefill lo acepte.

Leído el kernel, FP32 no está en el algoritmo, está en cuatro sitios:

* dos guardas de dtype (`_impl.py:64`, `_tuning.py:105`);
* el **tipo del puntero** del estado, que en CuTe se deriva del binding: en el
  prefill estaba fijado a `Float32` a mano (`_cute_kernels.py:1945/1972`,
  `_parallel_kernels.py:457/458/480/484`), mientras el decode ya lo derivaba con
  `_numeric_type(binding.recurrent_state.dtype)`;
* el **store**: `_load_state` ya ensancha (`Float32(source[i])`, `_cute_kernels.py:
  1124-1138`) pero `_store_state` escribe el acumulador crudo (`:1118-1121`), y
  igual el commit de la ruta paralela (`_parallel_kernels.py:393/397`);
* el **stub de metadatos** con el que se compila por AOT
  (`delta_prefill/preparation.py:63`), que declaraba FP32 siempre.

O sea: cargar BF16 → operar FP32 → guardar BF16 es exactamente lo que el kernel
ya hace por dentro (el acumulador es FP32), más un tipo de puntero y una
conversión en el store. No hace falta reescribir el algoritmo, ni la toolchain,
ni el artefacto AOT: el parche cambia el texto antes de `vllm serve` y la
compilación AOT se hace sobre el texto parcheado en el mismo arranque. **Esto
contradice lo que afirmé aquí el primer día** («un parche de string no llega»):
no lo comprobé contra el código del kernel, lo inferí de las guardas. Está
arreglado en §5.1.

Queda una diferencia de fondo con A que el parche no borra: B2 cambia la
**precisión del prefijo** que el prefill escribe en el estado (BF16 en vez de
FP32), así que su greedy no es bit-idéntico a E2 ni a A. Cuánto se aleja es un
número de GPU (§5.1 y §7).

### 4.4 Qué hace el modo B del parche entonces

Aplica la relajación de selección (como A) más tres relajaciones condicionadas a
`external_records` (Caps de decode `:127-135`, init del kernel de replay
`_cute_kernels.py:364-368`, init del kernel de commit `:1316-1319`) y la de
`refuse_reasons` (`gdn_deferred_commit.py:120-124`, condicionada a
`QWEN_GDN_MIX=B` + `decode_kernel=="b12x"` + `compact_requested()`). Es **la
mitad de decode de B2** sin tocar el prefill, y sirve para medir B a c1 (greedy
de b12x decode con base BF16 + registros FP32 en lotes uniformes) y el
frío/caliente de la caché. Su límite es el que documenta §4.1/§4.2: aborta a
c2+. B2 (§5.1) es el modo que no aborta, porque el prefill también es b12x y no
hace falta levantar ni la selección ni `refuse_reasons`.

---

## 5. El entregable ejecutable (probado en CPU)

| fichero | qué es |
|---|---|
| `patch_gdn_mix.py` | 27 ediciones ancladas sobre 10 ficheros (`count(old)==1`), idempotentes por marca, `compile()` antes de escribir, **fail-closed en dos fases** (primero aplica y compila todo en memoria; si una ancla o un `compile()` falla, sale 3 **sin haber escrito nada**, aunque la edición rota sea de otro fichero). Cada edición declara los modos en que se aplica (`("A","B")`, `("B",)`, `("B","B2")`, `("B2",)`). Localiza la imagen con `importlib.util.find_spec("vllm")` — **no** `import vllm`, que inicializaría CUDA en el proceso padre antes de `vllm serve`. Con `QWEN_GDN_MIX` vacío no toca nada (exit 0). |
| `serve-gdn-mix.sh` | entrypoint: valida `QWEN_GDN_MIX`, corre el parcheador (aborta con 3 si falla), monta flags y env, `exec bash /etc/ursu/serve-${GDN_TP:-2}x.sh`. Con `QWEN_GDN_MIX=""` es un paso puro → reproduce el arranque de hoy (permite dejarlo montado permanente). |
| `prueba-gdn-mix.sh` | el arranque de la ventana (bloque probado de `prueba-e2.sh`) + confirmación por log + dos capturas greedy + compare + la prueba de aborto de B. Filtro por TP en el transform de `ursu.yaml` (probado en seco: genera 2 deploys para TP=2, 1 para TP=1, y el A0 con `EXTRA_ARGS` y diferidos a 0). |
| `greedy_compare.py` | `--capture` / `--compare` token a token a temperature 0 con `enable_thinking:false`; guarda los prompts en el JSON para que la segunda pasada use los mismos textos; casos: cortos, prefill largo troceado, frío/caliente de caché, turno 2 con acierto de prefijo, lote mixto a cN; informa `first_divergence`, `noise_floor` y `cold_vs_warm`, y marca `SUSPECTO`. |
| `test_patch_gdn_mix.py` | aplica el parcheador a una **copia** de `fork-src` (nunca al original): anclas casan, los 10 ficheros compilan (`py_compile`), idempotencia por modo, las marcas esperadas presentes y las ajenas ausentes (A no toca `b12x/` ni diferidos; A y B no arrastran ninguna de las 21 marcas de prefill de B2), modo vacío inerte, y fail-closed corruptiendo un ancla de A y otra de B2. **`RESULTADO: TODO OK`**. |
| `test_b2_gpu.py` | **prueba de kernel en GPU** (la corre una persona dentro de la imagen en un Spark; esta sesión no puede). Copia el `b12x` instalado a un tmp, le aplica B2 con prioridad en `sys.path` y comprueba que la carga del `b12x` cargado es la copia. Con formas reales (K=V=128; cabezas locales de TP=2 = 8/24 y TP=1 = 16/48, 2 secuencias, checkpoint a mitad) compara salida, estado final y checkpoints: b12x-FP32 vs oráculo FP32 (ancla absoluta) y **b12x-BF16 vs oráculo con el mismo estado BF16 de partida** (que es lo que prueba la conversión cargar-BF16→operar-FP32→guardar-BF16), más el dtype guardado y NaN/inf. Después **intenta** `algorithm=chunk_parallel` (pool FP32 + base BF16 por `_BoundaryKernel`/`_CommitKernel`); si la validación de config o la memoria compartida lo rechaza lo registra como `SALTO`, no como fallo. |
| `test_selection_logic.py` | extrae por AST la función `_resolve_gdn_backend_selection` **parcheada** y la ejecuta con stubs: `A/B + triton/b12x → ("triton","b12x")`; sin `QWEN_GDN_MIX` → `ValueError` (la política original intacta); `auto+b12x → ("b12x","b12x")`; `triton+triton → ("triton","triton")`; `flashinfer+triton` intacto; y la imagen **sin parchear** con `mix=A` sigue fallando. **`RESULTADO: LOGICA DEL SELECTOR OK`**. |
| `test_greedy_compare.py` | self-test del comparador contra un stub HTTP: capture/compare idéntico → exit 0; stub que cambia el primer token → todos `SUSPECTO` y exit 1; el suelo de ruido no silencia pero sí quita la sentencia por divergencia; y frío/caliente distinto → `SUSPECTO`. **`RESULTADO: COMPARADOR OK`**. |

Flags y env que quedan activos (los arma `serve-gdn-mix.sh`, no hay que
escribirlos en el yaml):

```
A: --mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel b12x
   --additional-config {"gdn_prefill_backend":"triton"}
   VLLM_GDN_DEFERRED_CHECKPOINTS=0  VLLM_GDN_COMPACT_RECORDS=0  QWEN_GDN_MIX=A
B: igual con VLLM_GDN_DEFERRED_CHECKPOINTS=1  VLLM_GDN_COMPACT_RECORDS=1  QWEN_GDN_MIX=B
B2: --mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel b12x   (SIN additional-config:
    con decode=b12x la política original ya resuelve prefill=b12x)
    VLLM_GDN_DEFERRED_CHECKPOINTS=1  VLLM_GDN_COMPACT_RECORDS=1  QWEN_GDN_MIX=B2
A0 (sin parche): --mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel cuda
   --additional-config {"gdn_prefill_backend":"triton"}  DEFERRED=0 COMPACT=0
```

### 5.1 B2: qué cambia cada edición y sus riesgos

Modo `B2` = mitad de decode de `B` (ediciones 4-6 y 6b) + mitad de prefill (7-25).
No toca la selección (1) ni `refuse_reasons` (3): con `prefill=b12x` y
`decode=b12x` la política original de `qwen_gdn_linear_attn.py:328-335` ya
devuelve `("b12x","b12x")` y `refuse_reasons` no añade motivo. Líneas de la
imagen **sin parchear** (`fork-src/.../dist-packages`).

| # | marca | fichero:línea | qué hace |
|---|---|---|---|
| 6b | `DECTUNE` | `b12x/sequence/gdn_decode/_tuning.py:155-163` | la validación de la consulta de preparación del decode admite base BF16 con registros externos, como 4-6. Faltaba: B2 no arrancaba («deferred GDN checkpoints require Qwen heads, FP32 state», 14-10 12:15Z). También en `B`. |
| 7 | `PREFCAPS` | `b12x/sequence/gdn_prefill/_impl.py:64` | Caps admiten `state_dtype` FP32 **o** BF16. |
| 8 | `PREFTUNE` | `b12x/sequence/gdn_prefill/_tuning.py:105` | el query de tuning deja de exigir `state_dtype=="float32"`. |
| 9 | `PREFPREP` | `b12x/sequence/_shared/delta_prefill/preparation.py:63` | el stub de metadatos del estado pasa a `getattr(torch, query.state_dtype)`. **Sin esto la AOT compila un kernel FP32 que luego no casa con el binding BF16** (`_recurrence_key` → `ValueError: compiled delta-rule recurrence kernel does not match the binding`). KDA sigue FP32: sus Caps no admiten BF16. |
| 9b | `PREFINNER` | `.../preparation.py:72` | el binding interno del `chunk_parallel` **declara FP32**: su `recurrent_state` es el pool de scratch, no el estado real. |
| 9c | `PREFPOOL` | `b12x/sequence/gdn_prefill/_parallel.py:47` | los Caps del plan interno llevan `state_dtype=torch.float32`, que es el dtype fijo del pool (`_parallel.py:57`); si no, `bind_tensors` rechaza el pool por dtype. |
| 10-11 | `PREFRT`, `PREFRTSTORE` | `b12x/sequence/_shared/delta_prefill/_cute_kernels.py:1029,1054` | `_RecurrenceKernel` recibe `state_type` (por defecto `Float32`, como el decode). |
| 12 | `PREFSTORE` | `.../_cute_kernels.py:1118-1121` | `_store_state` convierte al dtype del estado (`self.state_type(acc[...])`). `_load_state` ya ensanchaba, no se toca. |
| 13 | `PREFKEY` | `.../_cute_kernels.py:1891` | `binding.recurrent_state.dtype` entra en la clave del cache: un kernel FP32 no puede servir a un binding BF16 ni al revés. |
| 14-17 | `PREFCT`, `PREFCARG`, `PREFCFAKE`, `PREFCPTR` | `.../_cute_kernels.py:1914,1925,1945,1972` | `_compile_recurrence` deriva `state_type` del binding, se lo pasa al kernel, y el puntero del estado (fake de compile y launch real) usa `state_type` en vez de `Float32`. |
| 18-20 | `PREFCOMMIT*` | `b12x/sequence/gdn_prefill/_parallel_kernels.py:343-345,393,397` | `_CommitKernel` recibe `state_type` y convierte los dos stores al estado real (final y checkpoint desde el pool FP32). |
| 21-25 | `PREFAX*` | `.../_parallel_kernels.py:438,449,457-458,480,484` | `compile_auxiliary`: `state_type` en la clave, en el `_CommitKernel`, en los punteros fake y en los launches de `boundaries` (fuente = estado real, se lee y se ensancha) y `commit` (destino = estado real). El pool (`:453`) **se queda FP32**. |

Riesgos de B2:

* **El pool interno se queda FP32** (`_parallel.py:57`) y así lo declaran 9b/9c.
  Es scratch, no estado servible: su precisión no afecta al resultado final,
  pero ocupa `(slots, heads, 128,128)`×4 B en el scratch. Si algún día se
  quisiera BF16 ahí, es otro parche y otra conversación numérica.
* **`_BoundaryKernel` hereda de `_RecurrenceKernel`** pero sobreescribe
  `_store_state` con `st_global_v2_f32` (escribe solo al pool) y del estado real
  solo **lee**: por eso su `state_type` sigue siendo `Float32` y el cambio va en
  el tipo del puntero fuente (edición 24). Si alguien mueve la edición 12 a su
  override, corrompe el pool.
* **La precisión del prefijo cambia**: el prefill escribe el estado en BF16, así
  que el greedy de B2 no es bit-idéntico ni a E2 ni a A. No es un fallo de tipo
  si el error relativo se queda en la banda de BF16 (~0,2 % por valor); sí lo es
  si aparece en el primer token o se va a NaN.
* **La AOT y los caches persistentes**: `serve-2x.sh:1-2` siembra
  `$XDG_CACHE_HOME/b12x/compile/preparation` desde `/opt/b12x-seed/preparation`
  (solo JSON de preparación, no kernels) y `VLLM_USE_AOT_COMPILE=1` compila sobre
  el texto parcheado en el mismo arranque. Lo que protege de herir un kernel
  FP32 de un cache anterior es meter el dtype en la clave (ediciones 13 y 21):
  si no casara, `_recurrence_key` lanza `ValueError` al hacer launch — falla
  ruidoso, no sirve basura. Si el PVC de `XDG_CACHE_HOME` sobrevive entre
  experimentos, conviene limpiar `b12x/compile` al cambiar de modo.
* **GPU (14-10, dgx1)**: `test_b2_gpu.py` en verde con cabezas de TP=2 y TP=1, rutas
  `sequential` y `chunk_parallel`: salida, estado final y checkpoints dentro de la
  banda de BF16 (rel ≤ 0,94 %). Cubre el prefill; el decode con diferidos y base
  BF16 (ediciones 4-6b) solo lo cubre el greedy servido.

---

## 6. Cómo se corrompería el estado y cómo se detecta

| riesgo | mecanismo | cómo se ve |
|---|---|---|
| Layout de estado distinto entre el escritor triton y el lector b12x | el kernel b12x indexa `[cabeza][fila=valor][columna=clave]` y exige strides `(V*K, K, 1)` | **aborta**, no corrompe: `ValueError: recurrent state must be contiguous within each slot` (`_cute_kernels.py:1472-1477`) en el primer bind. Si salta, el análisis de §3.4 está mal. |
| dtype del estado no soportado por el kernel | Caps de decode | `ValueError` en construcción del layer (`gdn_decode/_impl.py:102-106`). |
| Política de selección | `:328-335` | `ValueError: b12x GDN prefill and decode must be selected together` → el pod no arranca. Es el fallo **esperado si el parche no corrió**: el pod se cae, no sirve E2 disfrazado. |
| Lote mixto con diferidos (solo B) | lee base incompleta | con compact: `RuntimeError: compact GDN records: spec-decode rows reached a non-b12x path` (`:3010-3014`). Sin compact (el caso que **no** implementé a propósito): sería silencio — registros perdidos → texto que se repite o se sale del tema a partir del primer lote mixto. |
| El decode b12x malinterpreta el estado que escribió el prefill | semántica, no layout | primer token generado distinto del de E2 (posición 0-1) y/o texto incoherente desde el inicio. El primer token depende **solo del prefill**, y el prefill de A es el mismo triton de E2 → divergencia en 0-1 = el decode leyó basura. Es el criterio `SUSPECTO` del comparador. |
| Caché de prefijos con estado incompleto | export/copy de bloque en modo align | `cache-cold` ≠ `cache-warm` en el mismo servidor (el comparador lo mide sin baseline). |
| Verificación MTP que reconstruye mal el estado | replay de `num_accepted_tokens` | desplome de `vllm:spec_decode_num_accepted_tokens_total / ..._draft_tokens_total` fuera de la banda de E2, o repetición. Se mira en `/metrics` antes y después. |
| NaN/inf | cualquier kernel | `grep -iE 'nan|inf|illegal memory|CUDA error'` en el log del pod. |
| Confundir ruido de batching con corrupción | troceo dependiente de reloj a c≥2 | **por eso** cada config se captura dos veces: `noise_floor` por caso. Sin eso, un `diverge@3` en un lote mixto no significa nada. |

---

## 7. Lo que NO he podido verificar sin GPU (marcado)

1. **A0**: si `fused_gdn_decode_post_conv_mtp` está compilado en su imagen. En
   la extracción no hay `.so` (el `RECORD` lo declara, el fichero no vino).
2. **Que el decode b12x sobre un estado escrito por el prefill triton dé texto
   correcto**. La evidencia de layout (§3.4) es fuerte y fallaría en el bind, no
   en silencio, pero la *semántica* del contenido (orden de cabezas tras el
   sharding TP=2, escala de `A_log`/`dt_bias`, conv post-conv) solo se prueba
   generando tokens: PLAN §2-3.
3. **Comportamiento real de B a c1** (greedy y frío/caliente con base BF16 +
   registros FP32) y el aborto esperado a c2. El guard está leído en el código,
   no ejecutado.
4. **Numérica de B**: con base BF16 el replay diferido **no es bit-idéntico** a
   la referencia FP32 que los guardas protegen. No es un bug, es otra
   aproximación; cuánto se aleja es un número de GPU.
5. **Perfil de CUDA graphs de A** (`UNIFORM_BATCH`): que las capturas terminen y
   no caiga a eager más que E2. Mismo valor teórico que E2, pero hay que verlo.
6. **Rendimiento**: si A (o A0) bate a E2 en tok/s de decode a 1/4/8. Sin eso el
   parche no paga la pena, por correcto que sea.
7. **El efecto de `VLLM_PREFIX_DROP_EXACT=1` + `align` + b12x decode** sobre los
   checkpoints de frontera en A.
8. Que `resolve()`/`resolve_compact()` acepten la configuración de B con las
   relajaciones puestas (el camino de `refuse_reasons` está leído, no ejecutado:
   requiere `use_v2_model_runner` — sí lo trae `env-2x.env:13` — y sin DP).
9. TP=1 (dgx3): todo lo anterior una vez más, que es donde su receta ya pide
   `DEFERRED=1` y `COMPACT=1` de casa.
10. **B2, entero**: las 21 ediciones de prefill están probadas en CPU (anclas,
    `py_compile`, idempotencia, marcas cruzadas, fail-closed) y el patrón de tipo
    está leído contra el decode, que ya hace cargar-BF16→operar-FP32→guardar-BF16
    (`gdn_decode/_cute_kernels.py:512`/`:699`/`:1531`). Falta la GPU, y son tres
    cosas distintas: (a) que el kernel de recurrencia de prefill dé el mismo
    resultado con base BF16 que el oráculo con esa misma base
    (`test_b2_gpu.py`); (b) que la **AOT** de `VLLM_USE_AOT_COMPILE=1` resuelva
    el artefacto BF16 en el arranque y no se lleve el FP32 por clave de cache;
    (c) que `chunk_parallel` sea siquiera configurable en `sm_121a` (memoria
    compartida) — si no lo es, la ruta empaquetada queda sin probar y B2 se sirve
    por `sequential`, que es el default de todos modos.

---

## 8. Recomendación

1. **A0 primero** (5 minutos, cero parche): si el op está y el greedy queda
   dentro del suelo de ruido, A0 es la respuesta y A solo se mide por tok/s.
2. **A** con `bash prueba-gdn-mix.sh A 2 $G/capturas/e2-1.json`: es la variante
   que el código soporta de verdad, con una sola relajación de política y el
   mismo comportamiento de lotes mixtos que E2 lleva sirviendo desde el arranque
   de la prueba.
3. **B**: medir solo a c1 y la prueba de aborto a c2; es el control que muestra
   por qué el prefill también tiene que ser `b12x`. **B2** es la línea servible y
   ya está escrita: `python3 -I test_b2_gpu.py` dentro de la imagen en un Spark
   (prueba de kernel) y, si sale OK, `bash prueba-gdn-mix.sh B2 2
   $G/capturas/e2-1.json`. Con B2 el paso 8 (aborto) no aplica: los lotes mixtos
   van por `b12x_mixed`. Si Ursu prefiriera no llevar parche en el arranque, el
   camino es upstream: derivar `state_type` del binding en el prefill, igual que
   hace el decode — son ~15 líneas en dos ficheros de `b12x`.
4. Si A se queda: la relajación de selección **es un flag de entorno acotado a
   este experimento** (`QWEN_GDN_MIX`), no una puerta de producción. Para
   servirlo de verdad hay que decidir si se vive con el parche en el arranque
   (como los de `launch-ursu-b12x.sh`) o se le pide a Ursu un upstream de la
   política en `:328-335`.
