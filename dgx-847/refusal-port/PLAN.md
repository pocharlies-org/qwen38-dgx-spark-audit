# DGX-847 — el dial de refusal sobre el fork de ursuciprian

Preparado, **no aplicado**. Nada tocado en el cluster ni en los Sparks: no hay ni un `apply`, ni un `patch`, ni una
petición a los servidores que se están midiendo. Lo que hay aquí son ficheros listos para ConfigMap y un plan de
ventana.

## Qué es

Nuestro método de abliteration (dirección rank-1 de refusal restada a las activaciones en runtime, con el dial
`λ` por petición detrás del alias `-uncensored`) sobre la imagen de ursuciprian
(`ghcr.io/ursuciprian/spark-vllm-b12x`, vllm `20260926.dev28+g884b4ff68.d20261002`, kernels b12x). Su receta y sus
kernels se quedan como están; aquí solo se enchufa el dial encima.

## Qué cambia respecto a nuestro mod de producción

| | nuestro mod (nightly RadixArk) | este port (fork b12x) |
|---|---|---|
| módulo del modelo | `models/qwen4_exp/nvidia/` | `models/qwen3_8_flash_next/` — **todas las anclas re-ubicadas** |
| overlays | 5 ficheros sustituidos (QSA, PLE, interface, modelopt) | **ninguno**: el fork trae lo suyo, aquí solo se parchea |
| proyecciones | 4 en el target + 3 en el borrador | 4 en el target (embed, ple, attn_out, mlp_out) + 2 en el borrador (embed, feedback) |
| prólogo MTP | dos lineales de Python, dos proyecciones | un kernel fusionado b12x (`mtp_feedback`): **una** proyección sobre la salida del kernel, en espacio hc |
| caché de compilación | torch.compile en caliente | artefacto AOT sembrado (`VLLM_USE_AOT_COMPILE=1` + `MEGA_AOT_ARTIFACT=1`): ver «AOT» abajo |
| recorte de prefill | no existía | `output_indices`: la proyección de attn va **antes** del recorte, si no las filas de `λ` se desalinean |
| clave del KV | guard `cache_salt is not None` | ese guard ya no existe; el `λ` entra en `extra_keys` (`cache_salt_keys`) |
| `NewRequestData` | ancla directa | gano tres kwargs: ancla re-ubicada por `boundary_checkpoint` |
| entrada del borrador | pegada a `propose()` | el bloque del boundary checkpoint se metió en medio: ancla re-ubicada (10 líneas, no 4) |

## AOT: por qué el hook no queda invisible y qué invalida la caché

El parche inserta líneas dentro de funciones compiladas, así que cambia su `co_firstlineno`. La clave del artefacto
AOT lleva ese número y vLLM, al cargar, verifica además el checksum de la fuente (`_verify_source_unchanged`,
`compilation/decorators.py:293`). Con el parche aplicado **la clave no casa y recompila**: el hook entra en el grafo
solo. Lo que hay que dejar como está:

- `VLLM_FORCE_AOT_LOAD=0` (default del fork). Con `1`, un mismatch de fuente es un error duro en vez de una
  recompilación; `install.py` aborta si viene a `1`, para que no se descubra a las 4 de la mañana.
- Consecuencia real y medida: los arranques de hoy cargan de disco (`compilation: 0.97 s`, init engine **319 s** en
  2× y **208 s** en 1×). El primer arranque parcheado paga la compilación entera y no hay medida suya; el segundo
  vuelve a estar caliente porque el artefacto nuevo se guarda en el hostPath `/srv/ursu-cache/{2x,1x}`.
- Y el guard a prueba de esto: con artefacto cargado de disco no hay trazado en el proceso, así que el guard de
  conteo (`verify_consumed`) leería cero sitios y mataría un servidor sano. El parche usa `verify_declared`
  (declaraciones del `__init__`, siempre eager) cuando `was_aot_compile_fn_loaded_from_disk` es cierto y
  `verify_consumed` en cualquier otro caso.

## Por petición

Igual que en producción, sin tocar LiteLLM: el alias `qwen38-flash-next-uncensored` manda
`extra_body.cache_salt = "refusal:1.0"`; `parse_request_lambda` lo lee en el scheduler, viaja en
`NewRequestData.refusal_lambda`, y el runner lo escribe en el buffer por token que lee el CUDA graph (tensor mutado
in-place; un float de Python se hornearía en la captura). El dial global sigue en
`POST /admin/refusal_lambda` (`collective_rpc`, fan-out a los dos ranks). Sin sello, manda el global.

**Pendiente de la ventana, y es el riesgo principal:** el borrador no lleva sello propio — se neutraliza al global
antes de entrar. Con global `λ=0` y una petición sellada a `1.0`, el drafter propone sobre la distribución *con*
refusal y la acceptance de k=4 cae. Se mide (aceptación con sello vs sin sello, y con global 0 vs global 1.0) y se
decide; la alternativa (sello también al drafter) es un cambio de semántica, no un arreglo.

## Cómo se instala en su contenedor

Mismo procedimiento que `install.py`/`run.sh` de producción, adaptado: se monta el mod en el pod y el entrypoint
instala y luego cede a la receta. Nada se hornea en una imagen (la imagen es suya).

ConfigMaps nuevos en `llm` (los `.npy` van solos a `binaryData`, `kubectl` lo hace solo con `--from-file`):

```bash
cd ~/bench-results/dgx-847/refusal-port
kubectl -n llm create configmap qwen38fn-b12x-root --dry-run=client -o yaml \
  --from-file=install.py --from-file=run.sh --from-file=base-hashes.json | kubectl apply -f -
kubectl -n llm create configmap qwen38fn-b12x-refusal --dry-run=client -o yaml \
  $(for f in refusal/*.py refusal/*.npy; do printf -- '--from-file=%s=%s ' "$(basename $f)" "$f"; done) | kubectl apply -f -
kubectl -n llm create configmap qwen38fn-b12x-refusal-payload --dry-run=client -o yaml \
  $(for f in refusal/payload/*.py; do printf -- '--from-file=%s=%s ' "$(basename $f)" "$f"; done) | kubectl apply -f -
```

En los tres Deployments de `ops/ursu.yaml`, y solo en ellos:

```yaml
args: [bash, /opt/ursu-mod/run.sh, /etc/ursu/serve-2x.sh]   # serve-1x.sh en el 1x
volumeMounts:
  - {name: b12x-mod, mountPath: /opt/ursu-mod, readOnly: true}
  - {name: b12x-refusal, mountPath: /opt/ursu-mod/refusal, readOnly: true}
  - {name: b12x-refusal-payload, mountPath: /opt/ursu-mod/refusal/payload, readOnly: true}
  - {name: probe-cal, mountPath: /opt/refusal-probe-cal, readOnly: true}   # solo si VLLM_REFUSAL_PROBE=1
volumes:
  - {name: b12x-mod, configMap: {name: qwen38fn-b12x-root}}
  - {name: b12x-refusal, configMap: {name: qwen38fn-b12x-refusal}}
  - {name: b12x-refusal-payload, configMap: {name: qwen38fn-b12x-refusal-payload}}
  - {name: probe-cal, configMap: {name: qwen38fn-refusal-probe-cal}}       # ya existe
```

y en `ursu-env-2x` / `ursu-env-1x`:

```
VLLM_REFUSAL_DIRECTION=/opt/refusal/refusal_direction_flashnext.npy
VLLM_REFUSAL_LAMBDA=0                       # global apagado; el sello por petición manda
VLLM_REFUSAL_PROBE=1                        # solo la pasada de calibración; apagado para medir velocidad
VLLM_REFUSAL_PROBE_LAYERS=16,20,24,28,32
VLLM_REFUSAL_PROBE_CAL=/opt/refusal-probe-cal/cal.json
```

`run.sh` ejecuta `install.py` y después `exec bash /etc/ursu/serve-2x.sh`. Si el mod no aplica, el pod muere ahí y
no sirve sin dial: fail-closed. `install.py` es idempotente (`/opt/refusal/b12x-installed.json`).

## Riesgos

1. **Hashes de la base.** `base-hashes.json` está fijado contra el árbol extraído de la imagen del window
   (`8abb60ee…`, 2×; el banner del 1× lleva la misma versión). Si se cambia de imagen sin regenerar los 16 hashes,
   `install.py` aborta con nombre y apellidos. Es el comportamiento querido, pero hay que saberlo.
2. **Semántica del λ en el drafter** (arriba): puede costar acceptance y por tanto tok/s. Es lo primero que se mide.
3. **Primer arranque parcheado = recompila.** No confundir ese coste con el del dial: se mide el arranque aparte.
4. **El checkpoint es otro trunk** (abajo): la dirección vale como punto de partida, no como valor final de λ.
5. Los pods corren `privileged` sin `runAsUser`, así que escribir en `dist-packages` y `/opt/refusal` funciona; si
   alguna vez corren como usuario, el instalador necesitaría `REFUSAL_ROOT` y un sitio escribible (no previsto).

## El checkpoint y la dirección

`ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` (rev `16c9bd54…`) **declara la misma base**
(`Qwen/Qwen3.8-Flash-Next`) y la misma geometría que nuestro residente: `hidden=2560`, 48 capas, vocab 248320,
`hc_count=4`, PLE en la capa 1, 1 capa MTP. La tabla de embeddings y `lm_head` son **byte-idénticas** a las del
NVFP4 de RadixArk (rev `7b719225…`, la que sirve nuestro residente). La dirección vive en el espacio de las
activaciones de esos módulos, así que **sigue siendo válida como punto de partida**.

No es el mismo modelo entero: el trunk está requantizado (NVFP4/MXFP8) y `ple.value_proj` y `mtp.fc_embedding` /
`mtp.fc_hidden` difieren (un QAD corto después de la cuantización). Por eso λ se re-confirma con la sonda en la
ventana, no se da por bueno. Re-extraer por diferencia de pesos **no se puede** sobre su checkpoint (no existe un
gemelo abliterated suyo); el plan B, si la sonda dice que λ=1.0 se ha corrido, es extracción en el residente
(DGX-442), que ya está escrita en `refusal/calibration/collect.py`.

## Qué se probó en CPU (x86, sin GPU) y con qué resultado

Se secó todo el instalador contra una copia limpia del árbol del fork: `install.py` completo (gate de hashes,
copia, tres parches, marker) aplicado y **verificado idempotente en segunda pasada**. 8 de los 9 gates CPU pasan,
0 fallos:

- `test_projection_equivalence` — el álgebra que sostiene el parche (proyectar la suma == ortogonalizar cada
  writer): err 3.8e-06 a 1e-05, `embed` exacto, `project_hc` 1.2e-07.
- `test_per_request_lambda` — sello → buffer por token, tramo `λ=0` bit-exacto, tramo sellado == álgebra del autor.
- `test_hc_shape_contract` — coseno de la suma de los 4 streams con el tensor intacto; el borrador no pisa la
  capa 16 del target.
- `test_lambda_zero_identity`, `test_refusal_probe` (26 casos de la sonda: calibración, nan/inf, desactivación).
- `check_injected_scope` — los 3 ficheros con código inyectado: todos los nombres ligados (el `NameError` que
  tumbó el residente el 03-09 no puede repetirse).
- `check_guard_counts` — **98 sitios registrados** en la topología real (48 attn + 48 mlp + 1 ple + 1 embed) y las
  3 cuentas del guard casan también con una topología de control de 4 capas.
- `check_lambda_survives` — el runner llama al estado 5 veces, nada arrasa el buffer después de `fill_target`, y
  la neutralización del borrador está pegada a `speculator.propose()` (cota ampliada de 8 a 12 líneas por el
  bloque del boundary checkpoint del fork).

Falta `test_cache_salt_contract`: importa `vllm.v1.worker.gpu_worker` de verdad, así que solo corre dentro de la
imagen (es el gate del Dockerfile en producción). Su parte de parser ya está cubierta por `test_per_request_lambda`
con los literales exactos (`refusal:1.0`, `refusal:0`, basura alrededor, fuera de cota).

## Pendiente de la ventana

La ventana no tiene hora fija de cierre: termina cuando acabe la batería (benchy + quality en 2× y 1×, re-arrancada
a las 20:44Z). Orden y coste:

1. montar los 3 ConfigMaps y los args de `ursu.yaml`, arrancar **2×** parcheado → ver en el log
   `[b12x-rank1]`, `[qwen38next-runtime]`, `[refusal-probe]`, `b12x refusal mod applied`, y el arranque con
   recompilación (una vez, ~decenas de minutos; medir aparte).
2. sonda con `VLLM_REFUSAL_PROBE=1`: 20 prompts × {sin sello, `refusal:0`, `refusal:1.0`} → confirmar que λ=1.0
   sigue siendo el punto sobre **su** trunk (minutos).
3. velocidad: la misma batería dos pasadas, `λ=0` global y alias sellado, frente a la receta sin parche. Aquí sale
   el coste del dial y la aceptación de k=4 (lo que tarde la batería, ×2).
4. quality: la batería de quality que ya se usa, con el alias `-uncensored` apuntando al servidor parcheado.
5. si λ se ha corrido: extracción en el residente con `refusal/calibration/collect.py` y una pasada más corta.
