# PLAN-GPU · DGX-852 — validar `QWEN_GDN_MIX` (A) y (B) frente a E2

**Lo ejecuta la sesión dueña de los Sparks.** Esta sesión (DGX-852) no toca el
clúster: deja el parcheador, el entrypoint, el comparador y este plan.

Todo lo de aquí abajo son HTTP contra un endpoint y `kubectl` de la ventana de
prueba `llm/ursu-*` (etiquetas `dgx-847=prueba`, ningún Service de producción
las selecciona). El residente queda a 0 y el árbitro en pausa mientras dura la
prueba, como en `ops/prueba-e2.sh`.

```bash
export KUBECONFIG=~/.kube/config
kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'   # phase=ready antes de crear carga
```

Ficheros del experimento (esta carpeta):

| fichero | qué es |
|---|---|
| `patch_gdn_mix.py` | parcheador anclado, fail-closed, modos A/B (se ejecuta dentro del contenedor antes de `vllm serve`) |
| `serve-gdn-mix.sh` | entrypoint: parchea, arma flags/env y exec `serve-2x.sh` / `serve-1x.sh` |
| `prueba-gdn-mix.sh` | lanza A0/A/B en el TP pedido, confirma el selector por log y captura el greedy dos veces |
| `greedy_compare.py` | `--capture` / `--compare` token a token, con suelo de ruido y prueba frío/caliente |
| `test_patch_gdn_mix.py`, `test_selection_logic.py`, `test_greedy_compare.py` | los tres tests de CPU (pasan; ver INFORME.md §5) |
| `test_b2_gpu.py` | **prueba de kernel en GPU** del prefill `b12x` con estado BF16 (B2). Se copia el `b12x` del contenedor a un tmp, se le aplica B2 y se compara con el oráculo. Paso previo obligatorio a lanzar B2 (§4.5). |

---

## 0. Por qué se mide así (una párrafo, importa para leer los números)

Solo hay **un** 2x (dgx2+dgx1). Dos servidores con distinto TP no son comparables
token a token: el TP cambia el reparto de cabezas (`num_v_heads/tp`) y el orden
de los all-reduce, y eso domina la divergencia mucho más que el kernel GDN. Por
tanto **no se enfrentan E2 y A en paralelo**: se lanza E2, se captura, se
apaga, se lanza A en el mismo puesto y se compara contra el fichero. De ahí el
diseño `--capture` / `--compare` de `greedy_compare.py`.

Segundo punto: **el greedy no es reproducible entre ejecuciones bajo
concurrencia**. Con `--max-parallel-prefills 4` y `--prefill-policy
decode-aware`, donde caen las fronteras de troceo de un prefill largo depende
del reloj, y un troceo distinto da numeria distinta. Por eso cada configuración
se captura **dos veces**: `E2-1` contra `E2-2` es el suelo de ruido, y una
divergencia de A solo es interesante si supera ese suelo. El comparador informa
del suelo por caso (`noise_floor`) y no sentencia corrupcion por una
divergencia temprana en un caso que E2 ya no supo reproducir.

Tercero: hay una prueba que **no necesita baseline**, y es la más dura contra la
corrupción de estado: el mismo prompt largo dos veces seguidas en el mismo
servidor (`cache-cold` / `cache-warm`). La segunda pisa la caché de prefijo
alineada; si el estado guardado fuera distinto del que se computo en frio, la
salida cambia en los primeros tokens. Eso es corrupción, sin comparar con nada.

---

## 1. Fase 0 — A0: el decode CUDA fusionado, sin parche (5 min, se hace primero)

A0 = prefill triton + decode **cuda** (`fused_gdn_decode_post_conv_mtp`). La
política de vLLM **ya lo permite sin tocar nada**: el raise de
`qwen_gdn_linear_attn.py:328-335` solo se dispara cuando `prefill == "b12x" or
decode == "b12x"`; con `decode="cuda"` se llega a la línea 344 y devuelve
`("triton", "cuda")`. Requisitos del op en
`_fused_gdn_decode_unsupported_reason` (1409-1432): modelo BF16, K=V=128, conv
BF16, estado en `FUSED_GDN_STATE_DTYPES = (fp32, bf16)` (línea 218),
computabilidad ≥ 8.0 y `hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")`.

**El riesgo de A0 es el silencio**: si el op no está compilado y el kernel no se
pidió por `additional_config` de forma explícita, vLLM no aborta — loguea
`Falling back to the Triton GDN decode path: <razón>` (línea 819) y sirve E2
disfrazado, y el número que saques no significa nada. Compruébalo antes de
medir, de las dos maneras:

```bash
# a) ¿está el op en la imagen? (el fork extraído no trae .so, no se puede saber desde fuera)
kubectl exec -n llm deploy/ursu-2x-head -- python3 -I -c \
  "import torch, vllm._C; print('fused_gdn_decode_post_conv_mtp', hasattr(torch.ops._C,'fused_gdn_decode_post_conv_mtp'))"
# b) ¿dónde vive el .so si está? (RECORD del wheel lo declara)
kubectl exec -n llm deploy/ursu-2x-head -- sh -c \
  'ls -l /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so; \
   strings -a /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so | grep -c fused_gdn_decode_post_conv_mtp'
```

En la extracción local (`fork-src/.../dist-info/RECORD`) el fichero
`vllm/_C_stable_libtorch.abi3.so` (165 MB) está declarado pero **no vino en la
extracción**: no hay `.so` en `fork-src`, así que desde CPU no se puede
confirmar que el op exista. Ese es exactamente el dato que trae esta fase.

Si el op existe, lanza A0 y mide (no lleva parche; `prueba-gdn-mix.sh` le deja
los flags por `EXTRA_ARGS`, como `prueba-e2.sh`):

```bash
bash prueba-gdn-mix.sh A0 2
```

Los flags exactos de A0: `--mamba-ssm-cache-dtype bfloat16 --gdn-decode-kernel
cuda --additional-config {"gdn_prefill_backend":"triton"}` con
`VLLM_GDN_DEFERRED_CHECKPOINTS=0` y `VLLM_GDN_COMPACT_RECORDS=0` (los dos a 0
por obligacion: `gdn_deferred_commit.resolve()` es fail-closed y con
`decode != b12x` lanzaría `ValueError` — `refuse_reasons` 115-119 — y el env
`-2x` trae `DEFERRED=1` de casa).

**Decisión:** si A0 está compilado y su greedy contra E2 queda dentro del suelo
de ruido, A0 es la respuesta barata al pedido de Dani (decode no-triton con
prefill triton, cero parche) y A solo aporta si el kernel b12x de decode gana
tok/s medido. Si el op no está, A0 se descarta en dos minutos y se va a A.

---

## 2. Fase 1 — baseline E2 (la que ya está arriba, o relanzada)

Si E2 sigue arriba del run de las 08:59Z de `ops/prueba-e2.sh`, captura **ya**,
antes de tocar nada. Si no, relánzalo con ese script (deja el 2x en
`http://spark2:8000` y el 1x en `http://spark3:8000`) y luego:

```bash
G=/home/user/bench-results/dgx-847/mtp/definitivo/gdn-mix
mkdir -p $G/capturas
for n in 1 2; do
  python3 -I $G/greedy_compare.py --url http://spark2:8000 --model qwen3.8-flash-next \
    --long-tokens 16000 --conc 8 --capture $G/capturas/e2-$n.json > $G/capturas/e2-$n.log 2>&1
done
# control: E2 contra si mismo. Los casos que aqui ya divergen NO se pueden usar
# para sentenciar nada en las fases 2 y 3.
python3 -I $G/greedy_compare.py --url http://spark2:8000 --compare $G/capturas/e2-1.json \
  --noise $G/capturas/e2-2.json --out $G/capturas/e2-vs-e2.json
```

Los prompts del caso largo se calibran en la pasada `--capture` y **se guardan
en el JSON**: las pasadas `--compare` usan exactamente los mismos textos.

## 3. Fase 2 — variante A (decode b12x + prefill triton, sin diferidos)

```bash
bash prueba-gdn-mix.sh A 2 $G/capturas/e2-1.json
```

El script hace, en orden: residente a 0 + `drop_caches` hasta ≥100 GB libres en
dgx/dgx2/dgx3 (el bloque probado de `prueba-e2.sh:13-20`), recrea el ConfigMap
`ursu-recipe` con `serve-gdn-mix.sh` y `patch_gdn_mix.py` dentro, aplica
`ursu.yaml` filtrando solo el TP pedido con `args: [bash, /etc/ursu/serve-gdn-mix.sh]`,
`QWEN_GDN_MIX=A`, `GDN_TP=2`, espera `/health` (un reinicio = fallo, como en
`prueba-e2.sh:43-51`), comprueba el selector por log y captura dos veces.

**Lo que tiene que decir el log de arranque** (sin esto la prueba no vale):

| línea buscada | esperado |
|---|---|
| `QWEN_GDN_MIX=A aplicado: ...QWEN_GDN_MIX_SELECTION` | el parche entro |
| `GDN decode kernel: b12x` (`qwen_gdn_linear_attn.py:851`) | el selector acepto la mezcla |
| `Falling back to the Triton GDN decode path` | **ausente** (si aparece, no es A) |
| `VLLM_GDN_DEFERRED_CHECKPOINTS=1 but ...` / `ValueError` | **ausente** (A pide diferidos a 0) |
| `Capturing CUDA graph` / `Graph capturing finished` | sin errores; ver nota UNIFORM_BATCH abajo |

Env/args exactos que monta `serve-gdn-mix.sh` en modo A (el resto de la receta
de ursu no cambia):

```
--mamba-ssm-cache-dtype bfloat16  --gdn-decode-kernel b12x
--additional-config {"gdn_prefill_backend":"triton"}
VLLM_GDN_DEFERRED_CHECKPOINTS=0   VLLM_GDN_COMPACT_RECORDS=0
QWEN_GDN_MIX=A                    GDN_TP=2
```

(`--gdn-decode-kernel` acaba en `additional_config["gdn_decode_kernel"]` —
`arg_utils.py:2676-2677` — que es de donde lo lee el selector; por eso vale
tanto por CLI como por JSON.)

## 4. Fase 3 — variante B (A + registros compactos + checkpoints diferidos)

```bash
bash prueba-gdn-mix.sh B 2 $G/capturas/e2-1.json
```

Cambia solo `QWEN_GDN_MIX=B`, y con ello `VLLM_GDN_DEFERRED_CHECKPOINTS=1` y
`VLLM_GDN_COMPACT_RECORDS=1`. El log debe añadir
`GDN deferred checkpoints enabled` y `GDN compact records enabled`
(`gdn_deferred_commit.py:220-228` y `189-193`); si falta una de las dos, el
parche no hizo lo que se pedía y se para.

**Resultado esperado y buscado a propósito:** B levanta, sirve a **c1** y
**aborta en cuanto un lote mixto tiene filas especulativas**, con

```
RuntimeError: compact GDN records: spec-decode rows reached a non-b12x path
```

(`qwen_gdn_linear_attn.py:3010-3014`). Eso no es un fallo del parche: es el
fork diciéndonos que B1 (prefill triton + diferidos/compactos) no tiene camino
para lotes mixtos. El paso 8 de `prueba-gdn-mix.sh` lo fuerza a c2 al final,
porque tira el pod. **Si B NO aborta a c2, el hallazgo es otro** (o el guard no
se alcanzo, o el scheduler no mezclo nunca): repetir con `--conc 4` y mirar
`num_prefills`/`num_spec_decodes` en el log del step.

Lo que sí se puede medir de B en esta fase: greedy a **c1** contra E2 a c1
(`--conc 1`), que ejercita b12x decode con base BF16 + registros FP32 en lotes
uniformes, y el frío/caliente de la caché.

## 4b. Fase 3b — B2: prefill `b12x` + decode `b12x` con estado BF16

**Primero el kernel, después el servidor.** B2 toca el kernel CuTe de prefill; si
su numérica está mal, el greedy lo dirá pero no dirá dónde. `test_b2_gpu.py` se
juzga en 30 segundos y sin modelo cargado.

```bash
G=$D/mtp/definitivo/gdn-mix
IMG=$(kubectl get deploy -n llm ursu-2x-head -o jsonpath='{.spec.template.spec.containers[0].image}')
kubectl run b2gpu -n llm --image=$IMG --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"b2gpu","image":"'$IMG'","command":["sleep","infinity"],"resources":{"limits":{"nvidia.com/gpu":1}}}]}}'
kubectl wait --for=condition=Ready pod/b2gpu -n llm --timeout=180s
kubectl exec -n llm b2gpu -- mkdir -p /tmp/gdnmix
kubectl cp $G/test_b2_gpu.py    llm/b2gpu:/tmp/gdnmix/test_b2_gpu.py
kubectl cp $G/patch_gdn_mix.py  llm/b2gpu:/tmp/gdnmix/patch_gdn_mix.py
kubectl exec -n llm b2gpu -- python3 -I /tmp/gdnmix/test_b2_gpu.py
kubectl delete pod -n llm b2gpu          # solo este pod suelto, no un Application
```

El test **no toca el `b12x` instalado**: lo copia a un tmp, le aplica el modo B2
con `--root` y le da prioridad en `sys.path`. Sale `RESULTADO: B2 PREFILL BF16 OK`
o lista los FALLO/SALTO. Un `SALTO` de `chunk_parallel` no bloquea B2 (el default
de vLLM es `sequential`); un FALLO de `bf16 ... vs oraculo-bf16` sí: significa
que la conversión cargar-BF16→operar-FP32→guardar-BF16 no es correcta y no se
sirve.

Con el test en verde:

```bash
bash prueba-gdn-mix.sh B2 2 $G/capturas/e2-1.json
```

`serve-gdn-mix.sh` levanta `QWEN_GDN_MIX=B2` con `--mamba-ssm-cache-dtype
bfloat16 --gdn-decode-kernel b12x` **sin** `--additional-config`: con
`decode=b12x` la política original (`qwen_gdn_linear_attn.py:328-335`) ya
resuelve `prefill=b12x`, y `refuse_reasons` no añade motivo. Debe loguear la
línea `QWEN_GDN_MIX=B2 decode=b12x prefill=b12x ssm=bf16 deferred=1 compact=1`,
más `GDN deferred checkpoints enabled` y `GDN compact records enabled`.

Diferencias con la fase 3 (B), para leer el resultado:

* **el paso 8 de `prueba-gdn-mix.sh` (forzar el aborto a c2) NO corre**: está
  condicionado a `MODE=B`. Con prefill `b12x` los lotes mixtos van por
  `_forward_core_b12x_mixed`, que es el camino para el que existe el guard; si
  B2 aborta con `spec-decode rows reached a non-b12x path`, **el hallazgo es que
  `_can_use_b12x_gdn_decode`/`b12x_mixed` no se activaron** (mirar
  `num_prefills` y el log del builder de metadatos), no que falte el parche.
* **el greedy de B2 no se espera idéntico a E2 ni a A**: la base del estado es
  BF16 también en el prefill, así que el prefijo es otra aproximación. Criterio:
  `first_divergence > noise_floor`, texto fluido, `cold_vs_warm` `identical`, y
  la aceptación MTP en la banda de E2. Divergencia en posición 0 con NaN o
  repetición = el kernel lee mal la base.
* si `test_b2_gpu.py` no dio el `chunk_parallel` (SALTO), B2 se sirve por
  `sequential` igualmente; queda documentado que la ruta empaquetada no está
  probada en `sm_121a`.

## 5. Criterios de éxito

Para cada caso del JSON de compare (`first_divergence` = posición del primer
token distinto respecto a E2; `noise_floor` = en qué posición divergió E2 de E2
en ese mismo caso):

| caso | A: pasa si | B: pasa si |
|---|---|---|
| `short0..2` | `identical` o `first_divergence ≥ 32` con texto fluido; y `first_divergence > noise_floor` si el suelo existe | igual, a `--conc 1` |
| `long-chunked-prefill` (≈16k tokens > `--max-num-batched-tokens 8192` → 2+ trozos) | el **primer token generado coincide con E2** (depende solo del prefill, que es el mismo triton): divergencia en posición 0 = el decode leyó mal el estado; y `cold_vs_warm` `identical` | idem a c1 |
| `cache-cold` vs `cache-warm` | **`identical` o divergencia ≥ 32**: es la prueba interna de que el estado en caché es correcto, sin baseline | idem; si difiere, la caché con diferidos está incompleta → B no sirve |
| `prefix-cache-turn2` | el turno 2 (que pisa la caché alineada) no degenera y diverge tarde | idem a c1 |
| `mixed-c8-*` | ningún caso `SUSPECTO` y salida coherente en las 8 (aquí A usa triton para el decode: es el camino E2 ya probado) | **se espera el `RuntimeError` de arriba**; si aborta, B queda descartado para concurrencia ≥2 |

Además, en todas las fases:

* **NaN/inf**: `kubectl logs -n llm -l "$SEL" | grep -iE 'nan|inf|illegal memory|CUDA error'` vacío.
* **Aceptación MTP k=4** (el detector más sensible de un estado mal leído en la
  verificación): `curl -s $R/metrics | grep -E 'spec_decode_num_(accepted|draft)_tokens_total'`
  antes y después del barrido; `aceptados/borrador` debe quedarse en la banda
  de E2 (±0,05). Un desplome a ~0 con texto todavía fluido = el verificador
  reconstruye mal el estado y descarta borradores; un desplome **con** texto
  repetitivo = corrupción.
* **Herramientas**: los `tool_call` del fixture siguen siendo JSON válido
  (`qwen38_decode_sweep.py --quality --compare` ya lo comprueba y sale 1 si no).
* **Sin reinicios**: `restartCount` de los pods sigue en 0 al terminar.
* **Rendimiento** (no es el criterio de calidad, pero es la razón del
  experimento): el `mide` de `prueba-e2.sh:61-65` (`qwen38_decode_sweep.py` a
  1,4,8) con A debe batir a E2 en tok/s de decode; si no lo bate, A no merece
  el parche aunque sea correcto.

## 6. CUDA graphs: lo que hay que mirar en el arranque de A

Con prefill triton, `GDNAttentionBackend.get_cudagraph_support`
(`gdn_attn.py:158-172`) devuelve `UNIFORM_BATCH` en vez de `ALWAYS` (que exige
`prefill == decode == "b12x"`). Es **el mismo perfil de captura que tiene E2
hoy**, así que no se introduce riesgo nuevo, pero hay que confirmarlo en el log:
las capturas terminan (`Graph capturing finished`) y no hay caídas a eager por
`Failed to capture`. Si A capturase menos graphs que E2, el tok/s medido no es
comparable.

## 7. Réplica en TP=1 (dgx3) y cierre

```bash
bash prueba-gdn-mix.sh A 1 $G/capturas/e2-1x-1.json   # y B 1, y A0 1
```

El 1x vale la pena porque quita el all-reduce RoCE de en medio (una variable
menos) y porque su `env-1x.env` ya trae `DEFERRED=1` **y** `COMPACT=1` de casa:
es el puesto donde la intención de ursu está más cerca de lo que B intenta. Las
capturas de E2-1x se hacen igual que en la fase 1 contra `spark3:8000`.

**Cierre de la ventana**: devolver el clúster como estaba (`vuelta-residente.sh`
/ el hook de resume de `residente-lib.sh`), y dejar `ursu-recipe` **sin** los
dos ficheros nuevos o con `QWEN_GDN_MIX=""` en el ConfigMap: con valor vacío el
entrypoint es un paso puro (ni parche ni flags) y reproduce el arranque de hoy.

## 8. Qué se guarda como evidencia

`capturas/{e2,A,B,A0}-{1,2}.json`, sus `.log`, los `*-vs-e2.json` de compare,
`{A,B,A0}-arranque.log` y los `metrics` antes/después. El JSON de compare es el
entregable: trae por caso `first_divergence`, `noise_floor`, `cold_vs_warm` y
`suspects`.
