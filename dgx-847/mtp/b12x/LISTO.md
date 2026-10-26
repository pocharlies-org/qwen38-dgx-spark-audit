# DGX-852 · b12x 1.3.0 en el residente qwen38 TP=3, sin imagen nueva

**Listo y probado en CPU; nada aplicado.** Se puede instalar `b12x` en nuestra imagen oficial al
arrancar con una rueda pura Python de 1,5 MB: **las 11 dependencias que declara ya las trae la
imagen** (comprobado contra el snapshot de containerd del nodo). Con eso `--linear-backend b12x`
(GEMM densos: FP8 block-scaled, MXFP8, NVFP4-W4A4, MXFP4) está disponible hoy, con el checkpoint
y el reparto actual. **El MoE NVFP4 por b12x no: los dos backends b12x rechazan expert-parallel y
`moe_intermediate_size=640` no se parte en 3** — hace falta una vista con los expertos enrutados
rellenos a 768 (lo que hizo el fork de ursu). Está preparado y separado por llaves.

Todo en `/home/user/bench-results/dgx-847/mtp/b12x/`. Rama con el cambio, commiteada y sin push:
`/home/user/k8s/k8s-ai-pocharlies/.claude/worktrees/DGX-852-b12x` (`DGX-852-b12x`, desde
`origin/main`).

---

## 1. Dependencias: b12x 1.3.0 frente a la imagen del residente

Leídas del `METADATA` de la rueda (PyPI, `b12x-1.3.0-py3-none-any.whl`) y de los 254 `*.dist-info`
del snapshot de containerd del nodo (`imagen-distpackages.txt`, raíz
`…/io.containerd.runtime.v2.task/k8s.io/cac8791b…/rootfs/usr/local/lib/python3.12/dist-packages`).
Prueba: `prueba/test-dependencias.py` → **11/11 cumplidas, 0 que bajar**.

| pide b12x 1.3.0 | trae la imagen | |
|---|---|---|
| `torch>=2.12.0` | 2.13.0+cu130 | ok |
| `cuda-python` | 13.4.1 (`cuda/bindings/*.cpython-312-aarch64-linux-gnu.so`) | ok |
| `nvidia-cutlass-dsl==4.6.2` | 4.6.2 | ok |
| `nvidia-cutlass-dsl-libs-base==4.6.2` | 4.6.2 | ok |
| `nvidia-cutlass-dsl-libs-core==4.6.2` | 4.6.2 | ok |
| `nvidia-cutlass-dsl-libs-cu12==4.6.2` | 4.6.2 | ok |
| `nvidia-cutlass-dsl-libs-cu13==4.6.2` | 4.6.2 | ok |
| `apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8,!=0.1.8.post0` | 0.1.11 | ok |
| `rich>=13` | 15.0.0 | ok |
| `transformers>=4.51` | 5.17.0 | ok |
| `safetensors>=0.6` | 0.8.0 | ok |
| python `>=3.10,<4.0` | 3.12 | ok |
| *(extras `dev`)* | pytest, tomli | no aplican |

- La rueda es **puro Python**: `py3-none-any`, 250 ficheros, **ni un `.so`** (kernels CuTe-DSL +
  Triton que se JIT-ean). Por eso no hay que elegir wheel por plataforma: vale igual en aarch64.
- El gating propio de b12x (`b12x/_lib/gating.py`) solo pide tres cosas: **SM120 o SM121**
  (GB10 = 12.1 ✔), `nvidia-cutlass-dsl>=4.6.0` (4.6.2 ✔) y `triton` (3.7.1 ✔).
- `cutlass` no está en `dist-packages/cutlass` sino en `nvidia_cutlass_dsl/dsl_packages/cutlass`,
  añadido al `sys.path` por `nvidia_cutlass_dsl_packages.pth`. Los 8 módulos exactos que importa
  b12x (`cutlass.base_dsl.compiler`, `cutlass.cute.nvgpu.warp.mma`, `cutlass._mlir.dialects.cute`,
  `cutlass.utils.blockscaled_layout`, `cutlass.utils.blackwell_helpers`, `cutlass.cute.runtime`,
  `cutlass.pipeline`, `cutlass.utils.static_persistent_tile_scheduler`) están todos; `cutlass.pipeline`
  es paquete, no módulo suelto.
- Por eso el `pip install` es con **`--no-deps`**: sin esa bandera pip intentaría bajar torch
  (2,5 GB aarch64) y triton (1,1 GB) — y el pod no tiene red. Medido: `pip install --no-index
  --find-links wheels b12x==1.3.0` sin `--no-deps` falla con
  `No matching distribution found for torch>=2.12.0`; con `--no-deps` resuelve e instala.
- `triton` no se toca: es 3.7.1 y b12x no lo fija. `torch` tampoco: 2.13.0+cu130.

## 2. Wheelhouse offline

Una rueda y su suma. Nada más hace falta.

| dónde | ruta |
|---|---|
| x86 (aquí) | `/home/user/bench-results/dgx-847/mtp/b12x/wheels/` |
| dgx1 (nvidia-dgx), origen del NFS | `/home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/` |
| los tres pods | `/cache/huggingface/wheelhouse/b12x-1.3.0/` (montaje **solo lectura**) |

Los tres Deployment (`qwen38-flash-next-head`, `-worker`, `-worker-dgx3`) montan ya esa caché: el
head y el worker con el PVC `llm/hf-cache-nvidia-dgx-ro` (NFS `10.0.0.1:/home/dibanerz/.cache/huggingface`)
y dgx3 con `llm/hf-cache-dgx1-dgx3-llm-ro` (NFS `10.0.2.1`, **el mismo directorio exportado**); los
tres en `/cache/huggingface`, `readOnly: true`. `/etc/exports` de dgx1 lista spark2/.143 y
los GIDs RoCE de los tres Sparks, así que **copiar en dgx1 lo ven los tres rangos**.

**No copiado todavía** (es el paso 1 de la ventana). El comando:

```bash
ssh dgx 'mkdir -p /home/dibanerz/.cache/huggingface/wheelhouse'
rsync -av --chmod=a+rX /home/user/bench-results/dgx-847/mtp/b12x/wheels/ \
  dgx:/home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/
# comprobar desde el nodo que ya monta el NFS (sin tocar el residente):
sha256sum dgx:/home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/b12x-1.3.0-py3-none-any.whl
#   c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3
```

`--chmod=a+rX` importa: el contenedor corre como root (uid 0, `User: None` en la imagen) y el
directivo del export es `ro`; con los permisos por defecto de `rsync` la rueda se lee igual, pero
la lista `SHA256SUMS` y el árbol tienen que quedar `a+r` para el `pip` del pod.

## 3. `launch.sh`: qué se cambia y dónde está la verdad

**La verdad del arranque no está en `origin/main`.** El ConfigMap vivo
(`llm/qwen38-flash-next-vllm-launch`, clave única `launch.sh`) son **572 líneas**; en `main`
(`k8s/qwen38-flash-next-nvfp4-vllm.yaml`) son **257** — el script del TP=2, sin `--nnodes`, sin
`QWEN_NCCL_RING`, sin `QWEN_TP_SIZE`. El TP=3 vive en `origin/pin/qwen38-tp3-20261010` (578 líneas),
que es lo que renderiza la Application `ai` (`overlays/qwen38-tp3-ventana-20261010`,
`targetRevision=01f9b4d`); el clúster está **OutOfSync** con ese pin (le faltan las 6 líneas del
`--safetensors-load-strategy` de DGX-820) porque la ventana corre con auto-sync apagado.

Ficheros entregados:

- `bloque-b12x.sh` — el bloque, tal cual (217 líneas, comentado).
- `launch-vivo-con-b12x.sh` — **789 líneas**: el ConfigMap vivo + el bloque insertado justo antes
  de `exec taskset … vllm serve`. Este es el que se aplica a mano con `kubectl`.
- `launch-b12x.diff` — el diff unificado contra el vivo.
- En la rama: el mismo bloque insertado en el manifiesto de `main` (base pedida), con `bash -n` y
  `yaml.safe_load_all` en verde. **La PR del TP=3 tiene que ir sobre `pin/qwen38-tp3-20261010`, no
  sobre `main`** — o `main` tiene que alcanzar al pin antes.

### Las dos llaves

| env del Deployment | qué hace |
|---|---|
| `QWEN38_B12X=1` | instala la rueda desde el wheelhouse y añade `--linear-backend b12x`. **Funciona con el checkpoint y el EP=3 de hoy.** |
| `QWEN38_B12X_MOE=1` | además `--moe-backend b12x` y **quita `--enable-expert-parallel` y su `--all2all-backend`** del array de args. Pide una vista con `moe_intermediate_size` divisible por TP: lo comprueba en el `config.json` de `MODEL_PATH` y sale con 73 si no cuadra. |
| `B12X_WHEELHOUSE` | opcional, por defecto `/cache/huggingface/wheelhouse/b12x-1.3.0`. |

Vaciada la env, el siguiente arranque es el de siempre: no hay nada que revertir en el manifiesto.
Con la llave puesta, si el wheelhouse, el import o el gating fallan **sale con 73** en vez de
seguir: arrancar ~33 min para servir sin b12x (o reventar a mitad con `Failed to find a kernel`)
no es una entrega.

El bloque hace, en orden: rueda presente + sha256 contra `SHA256SUMS`; los `Requires-Dist` de la
propia rueda evaluados contra lo instalado (`packaging` 26.3 está en la imagen, no una lista
nuestra copiada en el script); `pip install --no-index --no-deps --find-links …  b12x==1.3.0`
(idempotente: si ya está en 1.3.0, no repite); import de los **seis** submódulos que
`vllm/utils/b12x.py:33-42` carga; `b12x._lib.gating.is_b12x()` y
`b12x.gemm.blockscaled.is_supported()` en esta GPU; y, si se pidió MoE,
`b12x.moe.fused_moe.is_supported()` más la divisibilidad del `config.json`.

### Qué selecciona b12x en NUESTRA vLLM (0.28.1rc1.dev628+g2a02f6efe)

`--linear-backend` y `--moe-backend` son flags reales de esta imagen
(`engine/arg_utils.py:1659-1664`), con `b12x` y `flashinfer_b12x` en sus Literals
(`config/kernel.py:120-146` MoEBackend, `:186-207` LinearBackend).

- **GEMM** (`model_executor/kernels/linear/__init__.py:246-252`): el backend `b12x` agrupa
  `B12xFp8BlockScaledMMKernel`, `B12xMxfp4LinearKernel`, `B12xMxfp8LinearKernel`,
  `B12xNvFp4LinearKernel`, `B12xTensorFP8ScaledMMLinearKernel`.
  - FP8 block-scaled: hoy el arranque loguea `Selected CutlassFp8BlockScaledMMKernel for
    QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear`; con la llave pasa a
    `B12xFp8BlockScaledMMKernel` (`scaled_mm/b12x.py:47`, `mm_block_fp8`). Pide activación
    `(1,128)`, peso `(128,128)` y entradas/salidas múltiplos de 128: con nuestra vista cuadra
    (2560, 6144, 640, 768, 9216, 12288, 15360).
  - MXFP8 (la familia densa del checkpoint de ursu: q/k/v/o, indexer, `in_proj_a/b`, expertos
    compartidos, visión): `_POSSIBLE_MXFP8_KERNELS[CUDA]` (`:515-526`) pone a
    `B12xMxfp8LinearKernel` el 4º; la llave lo sube a primero (`mxfp8/b12x.py:45`,
    `b12x.gemm.mxfp8_linear.mm`, empaqueta peso+escala en `process_weights_after_loading`).
  - NVFP4 W4A4 denso: `B12xNvFp4LinearKernel` (`nvfp4/b12x.py:47`, `mm_nvfp4`) está el 10º en
    auto; la llave lo pone primero.
- **MoE NVFP4** (`fused_moe/oracle/nvfp4.py:192-215`): hay **dos** backends b12x y no son el
  mismo. `FLASHINFER_B12X` (los CUTLASS de flashinfer) está fuera de la selección automática a
  propósito, texto del propio oracle: *"intentionally excluded from auto-selection until the
  upstream CUTLASS SM121 MMA op guard is resolved; use `moe_backend="flashinfer_b12x"` to opt in
  explicitly"*. `B12X` (**la rueda b12x, que es lo que pide el ticket**) tampoco entra en
  `AVAILABLE_BACKENDS`. Hoy salen `FLASHINFER_CUTLASS` (así lo loguea el residente:
  `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`). Hay que pedirlo:
  `--moe-backend b12x` → `map_nvfp4_backend` (`:158`) → `B12xExperts` (`fused_moe/b12x.py:200`), que admite
  `(nvfp4, nvfp4)` W4A4, `(nvfp4, mxfp8)` W4A8 y `(nvfp4, None)` W4A16 — los tres formatos que
  trae la vista de ursu, incluidos los expertos MTP.
- **GDN: nada que seleccionar.** `--gdn-decode-kernel` no existe en esta imagen y
  `VLLM_GDN_DECODE_KERNEL` solo admite `cuda|triton` (`envs.py:131`, `:1234-1236`). Se queda lo
  nuestro: **hoy el decode está en `cuda`** (`GDN decode kernel: cuda` en el log) y el prefill en
  triton por `VLLM_ADDITIONAL_CONFIG={"gdn_prefill_backend":"triton"}`; el estado SSM ya está en
  BF16 (`MAMBA_SSM_CACHE_DTYPE=bfloat16`). Si se quiere el decode en triton es
  `VLLM_GDN_DECODE_KERNEL=triton` en el Deployment, y es **otra medida**, no esta llave: no lo he
  metido en el bloque para no mezclar las dos cosas.
- **Lo que NO existe aquí** (son knobs del fork de ursu, no de nuestra imagen):
  `--load-format b12x`, `VLLM_B12X_NVFP4_MXFP8_CHECKPOINT`, `VLLM_MXFP8_LM_HEAD`,
  `VLLM_QWEN38_HC_MXFP8`, `VLLM_MTP_DRAFT_VOCAB`, `VLLM_ENABLE_ROCE_ALLREDUCE`, `B12X_AUTOTUNE`.
  De nuestra env b12x-only solo hay `VLLM_B12X_MOE_FP4_FORCE_A16` (`envs.py:193`, `:1639-1642`):
  fuerza activaciones BF16 en el MoE FP4; se puede poner a mano si se quiere probar W4A16 en los
  expertos enrutados.
- **Caché del JIT**: `B12X_COMPILE_CACHE_DIR=/tmp/b12x-compile`, que es el hostPath que ya montan
  los tres (`/home/user/.cache/vllm-tmp`, NVMe local de cada Spark, root-owned). Sin eso cae en
  `/root/.cache/b12x/compile` y se pierde en cada arranque. Es variable *operacional* para b12x
  (`_lib/compiler.py:1578-1593`), así que moverla no invalida la clave de caché. El warmup de las
  formas de serving lo hace vLLM solo (`warmup/b12x_warmup.py`, llamado desde
  `kernel_warmup.py:243`, con `is_device_capability_family(120)`).

## 4. Pruebas en CPU (hechas, sin GPU y sin tocar el clúster)

| prueba | resultado |
|---|---|
| `pip download --no-deps --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 312 b12x==1.3.0` | resuelve y baja `b12x-1.3.0-py3-none-any.whl` ✔ |
| `pip install --dry-run --no-index --no-deps --find-links wheels b12x==1.3.0` en venv 3.12 | `Would install b12x-1.3.0` ✔; la instalación real pasa ✔ |
| idem **sin** `--no-deps` | falla con `No matching distribution found for torch>=2.12.0` → confirma que `--no-deps` es obligatorio y que no se toca torch/triton |
| `prueba/test-dependencias.py` ( rueda vs 254 dists de la imagen) | **11/11 ok** |
| `prueba/test-imports.py` (grafo AST de los 244 `.py` del paquete) | terceros exigidos: `torch`, `cutlass`, `cuda`, `triton`, `safetensors`, `vllm`, `flashinfer` — los seis primeros presentes; `flashinfer` (0.6.18.post1) está; `vllm` es la imagen; `exllamav3_ext` solo en `run_trellis256_dense` (import `try` dentro de una función, camino EXL3/Trellis, no el nuestro) |
| `import b12x` y los 6 submódulos en un venv **sin ninguna dependencia** | `import b12x` ok; `attention.paged` y `gemm.blockscaled` ok (api perezoso); **`b12x._lib.intrinsics` falla con `ModuleNotFoundError: No module named 'cutlass'`** → por eso el paso 4 del bloque importa los seis: si cutlass se rompiera, vLLM pondría el backend a `None` y con `--linear-backend b12x` el arranque reventaría tarde |
| `bash -n` del `launch.sh` vivo, del vivo+bloque (789) y del de main+bloque (474) | los tres en verde; `yaml.safe_load_all` del manifiesto modificado: 7 docs ✔ |
| lógica bash del bloque con `python3` stub | llave apagada → ARGS intacto; `QWEN38_B12X=1` → `… --linear-backend b12x` con EP intacto; `+QWEN38_B12X_MOE=1` → fuera `--enable-expert-parallel` y `--all2all-backend`, dentro `--moe-backend b12x` ✔ |
| llave encendida sin wheelhouse | `QWEN38_B12X ABORTA: no hay b12x-1.3.0-py3-none-any.whl en /cache/huggingface/wheelhouse/b12x-1.3.0` y **rc=73** ✔ |

## 5. Aplicar a mano en una ventana (NADA de esto está ejecutado)

Orden: rueda → ConfigMap → env de los tres Deployment juntos (TP=3: los tres rangos tienen que
arrancar con el mismo `launch.sh`; si solo cambia uno, el que no cambie se queda esperando).
Arbiter ahora: `effective_mode: llm-tp`, `phase: ready`.

```bash
export KUBECONFIG=~/.kube/config

# 0) guardas: medición MTP terminada y el árbitro en llm-tp/ready
kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}{"\n"}'

# 1) wheelhouse a dgx1 (lo ven los tres rangos por el NFS que ya montan)
ssh dgx 'mkdir -p /home/dibanerz/.cache/huggingface/wheelhouse'
rsync -av --chmod=a+rX /home/user/bench-results/dgx-847/mtp/b12x/wheels/ \
  dgx:/home/dibanerz/.cache/huggingface/wheelhouse/b12x-1.3.0/

# 2) ConfigMap de arranque (la clave unica es launch.sh; el fichero ya lleva el bloque)
kubectl create configmap qwen38-flash-next-vllm-launch -n llm \
  --from-file=launch.sh=/home/user/bench-results/dgx-847/mtp/b12x/launch-vivo-con-b12x.sh \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl get cm -n llm qwen38-flash-next-vllm-launch \
  -o jsonpath='{.data.launch\.sh}' | grep -c QWEN38_B12X      # >0 = aplicado

# 3) activar en los tres rangos (esto reinicia el residente: ~33 min, ~20 sin log)
kubectl set env -n llm deploy/qwen38-flash-next-head \
  deploy/qwen38-flash-next-worker deploy/qwen38-flash-next-worker-dgx3 QWEN38_B12X=1

# 4) comprobar que la eleccion es b12x (no suponer: sale en el log de cada rango)
kubectl logs -n llm deploy/qwen38-flash-next-head --tail=-1 | grep -m20 -E \
  'QWEN38_B12X|for NVFP4 GEMM|Selected .*ScaledMMKernel|NvFp4 MoE backend|GDN decode kernel'
#   se espera: "QWEN38_B12X rueda ok", "dependencias ok (11)", "instalado b12x==1.3.0",
#              "imports ok (6)", "gating ok cap=(12, 1)",
#              "Selected B12xFp8BlockScaledMMKernel for …" (hoy es Cutlass),
#              "Using B12xMxfp8LinearKernel for MXFP8 GEMM" si la vista es la de ursu,
#              y el MoE siga en FLASHINFER_CUTLASS (la llave de MoE NO va con EP).

# 5) vuelta atras (instantanea, no hace falta tocar el manifiesto)
kubectl set env -n llm deploy/qwen38-flash-next-head \
  deploy/qwen38-flash-next-worker deploy/qwen38-flash-next-worker-dgx3 QWEN38_B12X-
```

Para el **MoE** hace falta, además de `QWEN38_B12X_MOE=1`, una vista con los expertos enrutados
rellenos 640→768 (el conversor de `../checkpoint/convert-ursu-nvfp4-tp3.py` ya tiene la mecánica de
`shared_in`/`shared_down`; hay que aplicarla a `w1/w2/w3` de los 512 expertos, peso y escala, y
dejar `--enable-expert-parallel` fuera). Con la vista actual el bloque **aborta con 73 y lo dice
en la línea**, no se cuelga a mitad de carga.

Y para que la Application `ai` no pelee con el cambio: o se desactiva su auto-sync en la ventana
(como está hoy, OutOfSync) o el `launch.sh` va por PR al overlay del pin antes de tocar el clúster.

## 6. Riesgos

1. **Un `can_implement` que no cuadra es un fallo duro, no una caída a otro backend.**
   `_resolve_backend_kernels` (`kernels/linear/__init__.py:364-392`) filtra la lista por el backend
   pedido; si la lista filtrada no puede implementar la capa, `init_*_linear_kernel` termina en
   `raise ValueError("Failed to find a kernel…")` y el motor no arranca. Con `--linear-backend b12x`
   la lista filtrada es **un solo kernel**. Las formas de nuestras dos vistas encajan (múltiplos de
   128/32), pero esto solo se cierra arrancando. Rollback: vaciar la env (minutos, no 33).
2. **Cambiar `linear_backend`/`moe_backend` cambia el hash de `KernelConfig`**
   (`config/kernel.py:309-325`), que entra en la clave del compile cache de vLLM y del autotune de
   FlashInfer: el primer arranque con la llave recompila, así que los ~33 min de siempre son más.
   No es un cuelgue (`gotcha_qwen38_arranque_33min`).
3. **Instalar la rueda activa un plugin de vLLM.** `b12x-1.3.0.dist-info/entry_points.txt`
   registra `[vllm.general_plugins] b12x_fp6 = b12x.integration.vllm.plugin:register_b12x_fp6` y,
   con `VLLM_PLUGINS` sin poner, vLLM carga **todos** los del grupo
   (`vllm/plugins/__init__.py:56-77`) en cada proceso. Lo que hace es registrar el método de
   cuantización `b12x_fp6` y añadir su clase a `linear.WEIGHT_LOADER_V2_SUPPORTED`: con nuestro
   checkpoint (`quant_method: modelopt`, `MIXED_PRECISION`) no interviene, y si el plugin fallara
   vLLM lo loguea y sigue (no es fatal). Hoy el único que registra ese grupo es la propia vLLM.
4. **El JIT dentro de un request vivo.** vLLM no llama a
   `b12x.freeze_kernel_resolution()`, así que una forma no vista en el warmup compila en caliente
   (es lo que el propio `b12x/__init__.py` documenta como riesgo). `b12x_warmup` cubre
   `max_num_batched_tokens` (4096), `max_num_scheduled_tokens` y los 19 tamaños de grafo
   (`[1,2,4,5,8,10,16,20,24,32,40,48,50,56,64,72,80,88,96]`), y b12x deduplica políticas; con MTP
   k=4 y `long_prefill_token_threshold=2048` no debería salir ninguna forma nueva, pero **no está
   medido**. Si aparece, se ve como un pico de latencia aislado en el primer request de esa forma.
5. **Sin `--no-deps` el `pip` intentaría tocar torch/triton** (y no hay red: fallaría, pero con
   `--index-url` interno reinstalaría). El bloque lleva `--no-deps` fijo; si alguien lo edita, esa
   bandera no se quita.
6. **El wheelhouse vive en la caché de modelos**, que es `Retain`/solo lectura y la llenan las
   vistas de pesos (dgx1: 389 G libres, 56 % usado). 1,5 MB no incomodan, pero **una limpieza de la
   caché HF se lleva el wheelhouse** y el arranque con la llave puesta sale con 73. Es intencionado
   (falla en voz alta), y la copia es un `rsync` de 1,5 MB.
7. **El ConfigMap aplicado a mano diverge del git** (y hoy ya diverge: el clúster va 6 líneas por
   detrás del pin `01f9b4d`). Si alguien sincroniza la Application `ai` mientras, el `launch.sh`
   vuelve al del pin **sin el bloque** y la llave puesta pasa a ser un `--linear-backend` sin rueda
   instalada → caso 1. O se va por PR al pin, o se mantiene el auto-sync apagado y se avisa.
8. **El MoE por b12x cambia el colectivo**: sin EP los expertos se parten por TP y el paso es
   all-reduce en vez de all2all, y `B12xExperts` no admite `expert_map` ni EPLB
   (`fused_moe/b12x.py:513`). Pesa lo mismo por rango (512×768/3 = 171×768), pero no es
   comparable 1:1 con la medida de hoy: es una vuelta de batería aparte, no un flag gratis.
   Consuelo: en el camino explícito el oracle no calla — `_return_or_raise` (`nvfp4.py:249-264`)
   lanza `NvFp4 MoE backend 'B12X' does not support the deployment configuration since <razón>`,
   con la razón que devolvió `is_supported_config`. Por eso el pre-flight del bloque puede
   quedarse en aviso y no en adivinanza: si sale ese mensaje, dice qué gate falló.
9. **Calidad**: cambiar los GEMM densos cambia el redondeo. La batería de calidad (7 suites,
   DGX-847) es la que dice si la combinación se aguanta; con `VLLM_REFUSAL_PROBE=1` calibrado sobre
   los pesos FP8 actuales, además, la sonda de refusal puede desfasar (ya estaba anotado en
   `../checkpoint/env-diff.md`).

## 7. Ficheros de este directorio

- `LISTO.md` — esto.
- `bloque-b12x.sh` — el bloque de `launch.sh` (217 líneas), comentado con las citas de la imagen.
- `launch-vivo-con-b12x.sh` / `launch-b12x.diff` — el ConfigMap vivo + bloque (789) y su diff.
- `wheels/` — `b12x-1.3.0-py3-none-any.whl` + `SHA256SUMS` (1,5 MB; sha256 `c97d88635521a7fdd4f6…`).
- `imagen-distpackages.txt` — las 254 distribuciones de la imagen, leídas del snapshot de containerd.
- `live/launch-vivo.sh` (572), `live/launch-pin.yaml` (578), `live/launch-main.yaml` (257) y
  `live/launch-main-con-b12x.sh` — las tres versiones del script, para no volver a adivinar cuál
  es la verdad, y el de `main` con el bloque (el que está commiteado).
- `inspeccion/b12x-1.3.0/` — la rueda desempaquetada (244 `.py`, `entry_points.txt`).
- `prueba/` — las pruebas de CPU: `test-dependencias.py` (rueda vs imagen), `test-imports.py`
  (grafo AST de importes), `salida-imports.txt` (su salida), `launch-filtro.sh` + `stub/python3`
  (la lógica bash de ARGS con `python3` simulado), `launch-sintactica.sh` + `bloque-codigo.sh`
  (el bloque pelado para `bash -n`), `venv-b12x/` (con dependencias) y `venv-vacio/` (sin ninguna:
  donde `import b12x._lib.intrinsics` revienta por `cutlass`).
- `img-config.json`, `manifest-arm64.json` — config y manifiesto arm64 de la imagen (capas, ENV,
  `VLLM_BUILD_COMMIT=2a02f6efe`).

Rama git: `DGX-852-b12x` en `.claude/worktrees/DGX-852-b12x`, **commiteada, sin push ni PR**.
