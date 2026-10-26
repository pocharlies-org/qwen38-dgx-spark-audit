#!/usr/bin/env bash
set -euo pipefail

# El enlace RoCE tiene DOS puertos. La receta de origen usa el f1
# (enp1s0f1np1 / rocep1s0f1); nuestro rdma-link-200g.service fija el f0.
# Se mantiene f0 porque es el que esta cableado y verificado a 200 Gb/s en
# este par de maquinas; cambiarlo seria otro experimento.
IFACE="${RDMA_IFACE:-enp1s0f0np0}"
HCA="${RDMA_HCA:-rocep1s0f0}"

est=$(cat /sys/class/infiniband/${HCA}/ports/1/state 2>/dev/null | head -1 || echo desconocido)
case "$est" in
  *ACTIVE*) echo "RDMA ${HCA} puerto 1: $est" ;;
  *) echo "ABORTA: ${HCA} puerto 1 no esta ACTIVE (state='${est}'). Cable DAC?" >&2; exit 7 ;;
esac

# Pinning de CPU. La receta de origen usa `taskset -c 5-9,15-19`: 10 cores
# concretos. En decode el cuello es el lanzamiento de kernels desde CPU, no
# la GPU, asi que esto no es cosmetico.
CPUS="${VLLM_CPUSET:-5-9,15-19}"

ARGS=(
  "$MODEL_PATH"
  --revision "$MODEL_REVISION"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 8000
  --tensor-parallel-size 2
  # Multinodo. La receta de origen NO lleva estos flags porque Sparkrun los
  # inyecta con su runtime `vllm-distributed`; al traducirla a k8s hay que
  # ponerlos a mano. Sin ellos vLLM aborta con "World size (2) is larger
  # than the number of available GPUs (1) in this node".
  # Los cuatro solo aplican con distributed_executor_backend=mp, segun la
  # docstring de vllm/config/parallel.py.
  --distributed-executor-backend mp
  # 01-09-2026: expert parallel. La receta de referencia de getrefined
  # (Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark) atribuye a EP el grueso de su
  # ventaja: "presumably EP plus vLLM's spec-decode-in-graphs being more
  # mature". Nosotros corriamos TP=2 a secas.
  #
  # Medido antes de esto, con SU metodologia (streaming, temp 0, decode
  # excluyendo TTFT, mediana de 2 reps):
  #   code 45.1 | reasoning 48.8 | math 51.3 | csharp 38.3 | prosa 48.1
  #   su mediana (code+reasoning+csharp) = 55.8; la nuestra 45.1  -> -19%
  # OJO: en prosa ya ibamos MAS rapido (48.1 vs sus 36.4). Su titular de
  # 55.8 es la mediana de code/reasoning/csharp, no incluye prosa, que es
  # su peor caso.
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --nnodes 2
  --node-rank "$VLLM_NODE_RANK"
  --master-addr "$VLLM_MASTER_ADDR"
  --master-port 29501
  --trust-remote-code
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
)
# 10-09-2026 (A/B equidad del prefill): --long-prefill-token-threshold T.
# vLLM V1 no tiene max_num_partial_prefills; T capa los tokens que UNA
# peticion puede meter en un paso (scheduler.py, dos sitios), asi que caben
# 4096/T prefills por paso y un "hola" deja de esperar minutos detras de un
# frio de 254K. No cambia el throughput total de prefill ni el paso (sigue
# eager con FULL_DECODE_ONLY). Vacio o 0 = sin tope (comportamiento de siempre).
if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && [ "${LONG_PREFILL_TOKEN_THRESHOLD}" != "0" ]; then
  ARGS+=( --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" )
fi
# 24-09-2026 (A/B opcion E, docs/qwen38-flash-next-cache-ab-20260924.md): cache de
# prefijos fina. PREFIX_MATCH_UNIT parte el hash de prefijos por debajo del bloque de
# 3200 (debe dividir 3200 y ser multiplo de 8); MAMBA_FINE_GRAINED_PREFIX_CACHE=1 guarda
# el estado Mamba en la union exacta del prefijo compartido entre conversaciones.
# Exige disable_eagle_block_drop:false y MTP de un modulo. Medido: peor que D en
# continuaciones; se deja el soporte APAGADO (vacio/0) como alternativa sin el flag.
if [ -n "${PREFIX_MATCH_UNIT:-}" ] && [ "${PREFIX_MATCH_UNIT}" != "0" ]; then
  ARGS+=( --prefix-match-unit "$PREFIX_MATCH_UNIT" )
fi
if [ "${MAMBA_FINE_GRAINED_PREFIX_CACHE:-0}" = "1" ]; then
  ARGS+=( --enable-mamba-fine-grained-prefix-cache )
fi
# 19-09-2026: --kv-cache-memory-bytes fija el pool KV en BYTES, igual en los dos
# rangos. Sustituye a perseguirlo con --gpu-memory-utilization, que NO
# sirve para igualarlo: el pool se calcula por rango desde la memoria libre
# LOCAL, asi que el mismo 0.8 daba 20,25 GiB en el head y 24,37 en el
# worker. Y el TP=2 usa el MINIMO, con lo que la diferencia del worker no
# la aprovechaba nadie.
#
# Se llama asi, con sufijo -bytes (el campo es CacheConfig.kv_cache_memory_bytes);
# `--kv-cache-memory` a secas NO existe y el motor muere con
# "unrecognized arguments". Ya lo usan vllm-qwen36-27b-uncensored y
# vllm-vision-deep. Con esto puesto, determine_available_memory() corre
# profile_run() y luego
# devuelve el valor TAL CUAL ("skipped memory profiling. This does not
# respect the gpu_memory_utilization config"). Consecuencia que hay que
# tener presente: la reserva pasa a ser CIEGA. Si el nodo no tuviese esa
# memoria, en vez de un fallo limpio al arrancar habria un OOM en runtime.
# Por eso GPU_MEM_UTIL se deja ajustado al total real de cada rango: ya no
# dimensiona el pool, pero request_memory() (v1/worker/utils.py) lo sigue
# usando como PUERTA de arranque —total*util frente a free—, asi que es lo
# unico que convierte una falta de memoria en un ValueError visible.
# Vacio o 0 = comportamiento de siempre (perfilado por utilizacion).
if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && [ "${KV_CACHE_MEMORY_BYTES}" != "0" ]; then
  ARGS+=( --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" )
fi
ARGS+=(
  # 01-09-2026: quitado --enable-prefix-caching, y resulto ser un NO-OP.
  # vLLM V1 lo activa POR DEFECTO: con el flag fuera, el engine sigue
  # reportando `enable_prefix_caching=True` y el arranque sigue diciendo
  # "Mamba cache mode is set to 'align' ... when prefix caching is enabled"
  # (config.py:605). Para apagarlo de verdad haria falta
  # --no-enable-prefix-caching, que NO se pone: el prefix caching es justo
  # lo que ayuda en carga agentica y multivuelta, que es el uso real aqui
  # (las sesiones de OpenCode reusan prefijos largos).
  #
  # Asi que la mejora medida en este paso (mediana 47.0 -> 47.2, csharp
  # 42.9 -> 44.8) viene de MAX_NUM_SEQS 10 -> 8, no de esto.
  # 01-09-2026: SIN --language-model-only. Al retirar DeepSeek, este pasa a
  # ser el backend de LITELLM_VISION_FALLBACK_MODEL, y con el flag puesto
  # LiteLLM desviaba las imagenes a un servidor que arrancaba ciego
  # ("All limits of multimodal modalities set to 0"). El model_info de
  # `qwen38-flash-next` ya declaraba supports_vision: true, asi que la
  # declaracion era falsa desde el 26-08.
  #
  # La torre visual NO cuesta NVFP4: quantization_config.ignore incluye
  # `model.visual.*`, y sus 333 tensores no llevan weight_scale -- van en
  # BF16. vLLM registra Qwen4ExpForConditionalGeneration en el registro
  # MULTIMODAL (vllm.models.qwen3_8_flash_next), no en el de texto.
  --compilation-config "$COMPILATION_CONFIG"
  # 01-09-2026: FUERA --no-enable-flashinfer-autotune. La receta de
  # referencia no lo lleva y con el arranque decia "Skipping FlashInfer
  # autotune" (kernel_warmup.py:170): los kernels de FlashInfer corren sin
  # tunear. El MoE va precisamente por FLASHINFER_CUTLASS (nvfp4.py:291),
  # asi que esto no es cosmetico.
  #
  # Venia de la epoca de SGLang, donde el autotune fue sospechoso del bug
  # del token 0. Ese diagnostico se FALSO midiendo (la causa real eran dos
  # condiciones cruzadas), y ademas era otro motor. Aqui la referencia
  # corre con autotune encendido y saca 55.8.
  #
  # REVERSION: volver a poner el flag. Base a batir sin el: mediana
  # (code+reasoning+csharp) 46.1 en frio / 45.6 tras soak; prosa 50.4.
  --speculative-config "$SPECULATIVE_CONFIG"
  ${VLLM_ADDITIONAL_CONFIG:+--additional-config "$VLLM_ADDITIONAL_CONFIG"}
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  # 02-09-2026: reportar la prefix cache en `usage`. El motor YA cachea
  # (94.5 % de hit en el log), pero sin este flag vLLM devuelve
  # `prompt_tokens_details: null`, LiteLLM omite el campo y opencode/
  # Paperclip cuentan cache_read=0: el run de STA-7 salio con 52.6 M de
  # "input tokens" que en realidad eran prefijo cacheado. Solo afecta al
  # JSON de respuesta, no al scheduling ni al KV.
  --enable-prompt-tokens-details
)


# 01-09-2026: el rank != 0 va HEADLESS. Sin esto el follower arranca un
# servidor completo -- API server + EngineCore propio -- y ese EngineCore
# llama a collective_rpc("get_kv_cache_spec"), que solo vale en el lider:
#
#   AssertionError: collective_rpc should not be called on follower node
#   (vllm/v1/executor/multiproc_executor.py:385)
#
# El follower moria con eso y el head se quedaba colgado en un colectivo
# NCCL esperando a un rank1 de otra generacion: 34 min sin una sola linea
# de log y sin reiniciar, porque el startupProbe tolera 120x30s = 60 min.
# No es el RDMA: los dos puertos estaban ACTIVE a 200 Gb/s.
#
# La receta de origen no lo lleva porque Sparkrun lanza el follower por su
# cuenta con el runtime `vllm-distributed`; al traducirla a k8s los dos
# Deployments ejecutan el MISMO script y hay que distinguirlos aqui.
#
# `--headless` pone api_server_count=0 (serve.py:65-72); con eso
# `run_headless` ve node_rank_within_dp > 0 y monta el MultiprocExecutor
# follower: la rama que el propio codigo comenta como "Run headless
# workers (for multi-node PP/TP)" (serve.py:213).
if [ "${VLLM_NODE_RANK}" != "0" ]; then
  ARGS+=( --headless )
  HEADLESS_ON=si
else
  HEADLESS_ON=no
  # 30-09-2026 (DGX-452): politica del scheduler SOLO en el rank 0. El
  # follower headless no planifica (no monta EngineCore propio), asi que el
  # flag no llega nunca a sus args aunque la env este en los dos
  # Deployments. Con "priority" el campo `priority` que inyecta el hook de
  # LiteLLM (priority_by_alias: -10 Hermes/Open WebUI, +10 hermes-batch)
  # decide el orden de admision; con fcfs se ignora. Menor = antes.
  # Vacio = sin flag (fcfs, el defecto de vLLM). Rollback: vaciar la env.
  if [ -n "${SCHEDULING_POLICY:-}" ]; then
    ARGS+=( --scheduling-policy "$SCHEDULING_POLICY" )
  fi
fi

# 09-09-2026 (test ursucipian): instalar el mod ANTES de arrancar. Los
# ConfigMaps se montan con symlinks (..data); se copian a un directorio
# limpio porque install.py hace copytree del bundle a /opt/refusal.
# install.py es FAIL-CLOSED: si la imagen no es la nightly 8a728663 (SHA de
# 13 ficheros base) aborta y el pod no arranca. Idempotente por marcador
# (/opt/refusal/ursucipian-installed.json). Corre los tests de CPU del dial
# de refusal en cada arranque de contenedor (unos minutos en arm64).
if [ -f /opt/ursu/install.py ]; then
  python3 - <<'PYEOF'
import pathlib, shutil
src, dst = pathlib.Path('/opt/ursu'), pathlib.Path('/tmp/ursu-bundle')
shutil.rmtree(dst, ignore_errors=True)
for p in src.rglob('*'):
    rel = p.relative_to(src)
    if p.is_file() and not any(part.startswith('..') for part in rel.parts):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst / rel)
PYEOF
  python3 /tmp/ursu-bundle/install.py
  echo "QWEN_URSU_MOD instalado (marcador $(ls /opt/refusal/ursucipian-installed.json 2>/dev/null || echo AUSENTE))"
fi
echo "QWEN_VLLM_EFFECTIVE rank=${VLLM_NODE_RANK} kv=${KV_CACHE_DTYPE} seqs=${MAX_NUM_SEQS} memutil=${GPU_MEM_UTIL} kvbytes=${KV_CACHE_MEMORY_BYTES:-0} headless=${HEADLESS_ON} v2runner=${VLLM_USE_V2_MODEL_RUNNER:-default} long_prefill_threshold=${LONG_PREFILL_TOKEN_THRESHOLD:-0} prefix_match_unit=${PREFIX_MATCH_UNIT:-0} mamba_fine_grained=${MAMBA_FINE_GRAINED_PREFIX_CACHE:-0} scheduling_policy=$([ "${VLLM_NODE_RANK}" = "0" ] && echo "${SCHEDULING_POLICY:-fcfs}" || echo n/a)"
# 08-10-2026 (DGX-672): FlashInfer CUTLASS con EP desigual (512 expertos / EP=3 =
# 171/171/170, vLLM los reparte desde 0/171/342). El binding de FlashInfer solo recibe
# ep_size/ep_rank y supone reparto igual, start = ep_rank * num_experts_on_rank
# (flashinfer_cutlass_fused_moe_binding.cu:397-399): el rango 2 empezaria en 340 y no en
# 342, y el kernel descarta los ids fuera de [start, end) (cutlass_fused_moe_kernels.cuh
# :438-441, :1970-1973, :2039-2041). Antes de llamar al kernel, los ids globales se pasan a
# la numeracion que el binding supone: slot local + start; un experto que no es de este
# rango (-1) cae en start-1 y el kernel lo descarta. ep_size/ep_rank no cambian: es el
# mismo camino que ya corre en TP=2. Solo actua con global_num_experts % ep_size != 0
# (inerte con reparto par); sin expert_map falla en voz alta en el primer forward. Sin
# sincronizacion GPU, compatible con la captura de CUDA graph. EPLB y expertos redundantes
# no estan activos; con ellos global_num_experts seria el numero de expertos fisicos y la
# condicion seguiria siendo coherente, pero no se ha probado. Idempotente por marcador.
# Si el patron no esta (imagen nueva) avisa y SIGUE sin parchear, y EP desigual saldria
# mal sin ruido: en cada arranque hay que ver "QWEN_FI_EP_UNEVEN (aplicado|ya aplicado)"
# en el log de CADA rango. Test: tools/test_qwen38fn_fi_ep_uneven.py (sobre una copia
# congelada del fichero de la imagen vllm 0.28.1rc1.dev628).
python3 - <<'PYEOF'
import pathlib, vllm
f = pathlib.Path(vllm.__file__).parent / 'model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py'
s = f.read_text()
old = "        _ = flashinfer_cutlass_fused_moe(\n"
new = ("        # QWEN_FI_EP_UNEVEN (DGX-672): reparto desigual de expertos entre rangos EP\n"
       "        if global_num_experts % self.ep_size:\n"
       "            if expert_map is None:\n"
       "                raise RuntimeError('QWEN_FI_EP_UNEVEN: expert_map ausente con reparto desigual')\n"
       "            _total = self.num_experts * self.ep_size\n"
       "            _start = self.ep_rank * self.num_experts\n"
       "            topk_ids = (expert_map[topk_ids] + _start) % _total\n"
       "            if not getattr(self, '_qwen_fi_ep_uneven_logged', False):\n"
       "                self._qwen_fi_ep_uneven_logged = True\n"
       "                print(f'QWEN_FI_EP_UNEVEN activo (E={global_num_experts}, ep={self.ep_size}, rank={self.ep_rank})', flush=True)\n"
       + old)
if 'QWEN_FI_EP_UNEVEN' in s:
    print('QWEN_FI_EP_UNEVEN ya aplicado')
elif s.count(old) == 1:
    f.write_text(s.replace(old, new))
    print('QWEN_FI_EP_UNEVEN aplicado')
else:
    print('QWEN_FI_EP_UNEVEN AVISO: patron no encontrado, sin cambios')
PYEOF
# 13-10-2026 (DGX-852): b12x SIN imagen nueva. `vllm[b12x]` es un extra QUE NOSOTROS DECLARAMOS
# (en el METADATA de nuestra vLLM 0.28.1rc1.dev628+g2a02f6efe pone Requires-Dist: b12x==1.3.0;
# extra == "b12x") y la rueda es puro Python: b12x-1.3.0-py3-none-any.whl, 1,5 MB, 250 ficheros,
# ni un .so (son kernels CuTe-DSL + Triton que se JIT-ean al primer uso). Se instala aqui, en el
# arranque, desde un wheelhouse en la cache NFS que los tres rangos ya montan en solo lectura
# (/cache/huggingface <- dgx1:/home/dibanerz/.cache/huggingface) y se apaga vaciando la env: el
# siguiente arranque vuelve a lo de siempre sin tocar el manifiesto.
#
# Por que cabe sin tocar nada mas: las 11 dependencias que declara b12x 1.3.0 YA ESTAN en la
# imagen, comprobadas contra el snapshot de containerd del nodo (13-10, 254 distribuciones):
#   torch>=2.12.0 ..................... 2.13.0+cu130
#   nvidia-cutlass-dsl==4.6.2 ......... 4.6.2 (y sus libs base/core/cu12/cu13, las cuatro 4.6.2)
#   apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8  0.1.11
#   cuda-python ....................... 13.4.1
#   rich>=13 / transformers>=4.51 / safetensors>=0.6 ... 15.0.0 / 5.17.0 / 0.8.0
#   python>=3.10,<4.0 ................. 3.12
# Y el gating propio de b12x (b12x/_lib/gating.py) solo pide tres cosas: compute capability
# 12.0 6 12.1 (GB10 = 12.1), nvidia-cutlass-dsl>=4.6.0 y triton (3.7.1 en la imagen). Por eso
# --no-deps: pip no toca torch ni triton, que es justo lo que romperia la imagen si resolviera
# por red (y el pod no tiene red).
#
# Dos llaves, porque las dos cosas que aporta tienen requisitos distintos:
#   QWEN38_B12X=1      instala y anade --linear-backend b12x. Cambia los GEMM densos que hoy
#                      elige 'auto': los FP8 block-scaled (en el arranque de hoy salen como
#                      "Selected CutlassFp8BlockScaledMMKernel for QKVParallelLinear /
#                      MergedColumnParallelLinear / RowParallelLinear") pasan a
#                      B12xFp8BlockScaledMMKernel, y MXFP8 / NVFP4-W4A4 / MXFP4 a sus kernels
#                      b12x (kernels/linear/__init__.py:246-252 y :441-448). Los W4A16 NVFP4 no
#                      tienen kernel b12x: _filter_kernels_by_backend (ibid.:347-361) avisa y
#                      sigue con la seleccion normal, asi que la GDN densa no cambia de backend.
#                      Funciona con EP=3 tal como esta hoy; no pide otro checkpoint.
#   QWEN38_B12X_MOE=1  ademas --moe-backend b12x y quita --enable-expert-parallel, porque
#                      B12xExperts._supports_parallel_config exige ep_size==1 y sin all2all
#                      (fused_moe/b12x.py:495-504) y FlashInferB12xExperts igual
#                      (fused_moe/experts/flashinfer_b12x_moe.py:199-203). Sin EP el MoE se
#                      reparte por TP y FusedMoEConfig.__post_init__ exige
#                      intermediate_size % tp == 0 (fused_moe/config.py:1346-1348): con
#                      moe_intermediate_size=640 y tp=3 no cuadra (640 % 3 = 1), asi que esta
#                      llave SOLO sirve con una vista con los expertos enrutados rellenos a 768
#                      (lo que hizo el fork de ursu; aqui se comprueba en el config.json de
#                      MODEL_PATH y se sale con 73 si no). Pesa lo mismo por rango que EP
#                      (512 x 768/3 = 171 x 768) pero el colectivo pasa a all-reduce.
#
# El GDN NO tiene ruta b12x en esta imagen: --gdn-decode-kernel no existe y VLLM_GDN_DECODE_KERNEL
# solo admite cuda|triton (envs.py:131 y :1234-1236). Se queda lo nuestro: decode cuda (lo que
# hay hoy), prefill triton por VLLM_ADDITIONAL_CONFIG y estado SSM en BF16 por
# MAMBA_SSM_CACHE_DTYPE. Tampoco existen aqui --load-format b12x,
# VLLM_B12X_NVFP4_MXFP8_CHECKPOINT, VLLM_ENABLE_ROCE_ALLREDUCE ni B12X_AUTOTUNE: los cuatro son
# knobs del fork de ursu. Si se quiere el decode en triton, es VLLM_GDN_DECODE_KERNEL=triton en
# el Deployment, y es otra medida, no esta llave.
#
# EFECTO SECUNDARIO de instalar la rueda: b12x trae un plugin de vLLM (entry_points.txt:
# [vllm.general_plugins] b12x_fp6 = b12x.integration.vllm.plugin:register_b12x_fp6) y, con
# VLLM_PLUGINS sin poner, vLLM carga TODOS los general_plugins (plugins/__init__.py:56-66) en
# cada proceso. Lo que hace ese plugin es registrar el metodo de cuantizacion `b12x_fp6` y anadir
# su clase a linear.WEIGHT_LOADER_V2_SUPPORTED: con nuestro checkpoint (quant_method modelopt,
# MIXED_PRECISION) no interviene, y si el plugin reventara vLLM lo loguea y sigue (no es fatal).
# Hoy el unico que registra ese grupo es la propia vLLM; con la rueda instalada son dos.
#
# La cache del JIT va al hostPath que ya es /tmp (NVMe local de cada Spark,
# /home/user/.cache/vllm-tmp): si no cae en /root/.cache y se pierde en cada arranque, y son
# ~20 tamanos de grafo x las capas del modelo compilando. B12X_COMPILE_CACHE_DIR esta en la lista
# de variables "operacionales" que b12x excluye de la clave de cache (_lib/compiler.py:1578-1593),
# asi que moverla no invalida nada. El warmup lo hace vLLM solo
# (model_executor/warmup/b12x_warmup.py, llamado desde kernel_warmup.py:243) sobre las formas de
# serving; no hay que darle nada.
#
# Rollback: QWEN38_B12X='' (y QWEN38_B12X_MOE=''). Con la llave puesta, si el wheelhouse o el
# gating fallan esto CIERRA con 73 en vez de seguir: arrancar 33 minutos para servir sin b12x (o
# reventar a mitad con "Failed to find a kernel") no es una entrega.
B12X_WHEELHOUSE="${B12X_WHEELHOUSE:-/cache/huggingface/wheelhouse/b12x-1.3.0}"
B12X_TP=$(printf '%s\n' "${ARGS[@]}" | awk 'take { print; take = 0 } $0 == "--tensor-parallel-size" { take = 1 }')
if [ "${QWEN38_B12X:-0}" = "1" ]; then
  B12X_WHEELHOUSE="$B12X_WHEELHOUSE" B12X_TP="$B12X_TP" B12X_MOE="${QWEN38_B12X_MOE:-0}" \
  MODEL_PATH="$MODEL_PATH" \
    python3 - <<'PYEOF' || { echo "QWEN38_B12X ABORTA (73): ver el mensaje de arriba" >&2; exit 73; }
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import zipfile

WHEELHOUSE = pathlib.Path(os.environ["B12X_WHEELHOUSE"])
WHEEL = "b12x-1.3.0-py3-none-any.whl"
# Los seis submodulos que vllm.utils.b12x importa al cargar (utils/b12x.py:33-42). Si uno no
# importa, vLLM lo trata como "b12x no disponible" y el backend pedido falla tarde y mal.
SUBMODULOS = (
    "b12x.attention.paged",
    "b12x.gemm.blockscaled",
    "b12x._lib.intrinsics",
    "b12x.gemm.mxfp8_linear",
    "b12x.gemm.tensor_fp8_linear",
    "b12x.moe.fused_moe",
)


def abort(motivo):
    print(f"QWEN38_B12X ABORTA: {motivo}", file=sys.stderr)
    sys.exit(1)


# 1. La rueda y su sha256 contra SHA256SUMS si el wheelhouse lo trae.
ruta = WHEELHOUSE / WHEEL
if not ruta.is_file():
    abort(f"no hay {WHEEL} en {WHEELHOUSE} (el NFS se monta de solo lectura: se copia en dgx1)")
sums = WHEELHOUSE / "SHA256SUMS"
real = hashlib.sha256(ruta.read_bytes()).hexdigest()
if sums.is_file():
    esperado = {p.split()[1]: p.split()[0] for p in sums.read_text().splitlines() if p.strip()}
    if esperado.get(WHEEL) not in (None, real):
        abort(f"sha256 de la rueda no cuadra: {real} != {esperado[WHEEL]}")
print(f"QWEN38_B12X rueda ok sha256={real[:16]}…")

# 2. Lo que DECLARA la rueda (METADATA dentro del .whl, no una lista nuestra copiada aqui),
#    contra lo instalado. Con --no-deps pip no baja nada, asi que si algo faltara hay que
#    decirlo aqui y no descubrirlo en el log del arranque.
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    abort("la imagen no trae packaging (hace falta para leer los Requires-Dist de la rueda)")
with zipfile.ZipFile(ruta) as z:
    meta = z.read("b12x-1.3.0.dist-info/METADATA").decode()
faltan, cumplidas = [], []
for linea in (l for l in meta.splitlines() if l.startswith("Requires-Dist:")):
    req = Requirement(linea.split(":", 1)[1].strip())
    if req.marker and not req.marker.evaluate({"extra": ""}):
        continue  # extras de desarrollo (pytest), no cuentan
    try:
        instalado = Version(md.version(req.name))
    except md.PackageNotFoundError:
        faltan.append(f"{req} (AUSENTE)")
        continue
    if req.specifier and not req.specifier.contains(instalado, prereleases=True):
        faltan.append(f"{req} -> instalado {instalado}")
    else:
        cumplidas.append(f"{req.name}={instalado}")
if faltan:
    abort("b12x pide y la imagen no cumple: " + "; ".join(faltan))
print(f"QWEN38_B12X dependencias ok ({len(cumplidas)}): " + ", ".join(cumplidas))

# 3. Instalar, o admitir que ya esta (el pod puede reiniciar dentro de la ventana).
if importlib.util.find_spec("b12x") is None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--disable-pip-version-check", "--quiet", "--find-links", str(WHEELHOUSE),
             "b12x==1.3.0"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        abort(f"pip install salio con {e.returncode}")
    print(f"QWEN38_B12X instalado b12x=={md.version('b12x')} desde {WHEELHOUSE}")
elif md.version("b12x") != "1.3.0":
    abort(f"b12x ya instalado, pero en otra version: {md.version('b12x')}")
else:
    print("QWEN38_B12X b12x==1.3.0 ya estaba instalado en este contenedor")

# 4. Que los seis submodulos que va a tocar vLLM importen.
for nombre in SUBMODULOS:
    try:
        importlib.import_module(nombre)
    except Exception as e:
        abort(f"import {nombre} falla: {type(e).__name__}: {e}")
print(f"QWEN38_B12X imports ok ({len(SUBMODULOS)} submodulos)")

# 5. El gating de b12x en ESTA GPU y, si se pide, la geometria del checkpoint para el MoE.
import torch
from b12x._lib.gating import is_b12x

if not torch.cuda.is_available():
    abort("sin CUDA visible: no se puede comprobar el gating de b12x")
if not is_b12x():
    abort(f"b12x solo corre en SM120/SM121 y esto es {torch.cuda.get_device_capability()}")
if not importlib.import_module("b12x.gemm.blockscaled").is_supported():
    abort("b12x.gemm.blockscaled.is_supported() es False en esta maquina")
print(f"QWEN38_B12X gating ok cap={torch.cuda.get_device_capability()} "
      f"cutlass-dsl={md.version('nvidia-cutlass-dsl')} triton={md.version('triton')}")

if os.environ.get("B12X_MOE") == "1":
    if not importlib.import_module("b12x.moe.fused_moe").is_supported():
        abort("b12x.moe.fused_moe.is_supported() es False en esta maquina")
    cfg = json.loads((pathlib.Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    moe_i, tp = int(tc["moe_intermediate_size"]), int(os.environ["B12X_TP"])
    if moe_i % tp:
        abort(
            f"QWEN38_B12X_MOE quita EP y entonces el MoE se parte por TP, pero "
            f"moe_intermediate_size={moe_i} no es divisible por tp={tp} "
            f"(fused_moe/config.py:1346). Hace falta una vista con los expertos enrutados "
            f"rellenos a {((moe_i + tp - 1) // tp) * tp}; esta es {os.environ['MODEL_PATH']}"
        )
    print(f"QWEN38_B12X MoE ok moe_intermediate_size={moe_i} % tp={tp} == 0")
PYEOF
  ARGS+=( --linear-backend b12x )
  if [ "${QWEN38_B12X_MOE:-0}" = "1" ]; then
    # b12x rechaza EP: fuera --enable-expert-parallel y su --all2all-backend del array.
    SIN_EP=()
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
      case "${ARGS[i]}" in
        --enable-expert-parallel) ;;
        --all2all-backend) i=$((i + 1)) ;;
        *) SIN_EP+=("${ARGS[i]}") ;;
      esac
    done
    ARGS=("${SIN_EP[@]}")
    ARGS+=( --moe-backend b12x )
  fi
  export B12X_COMPILE_CACHE_DIR="${B12X_COMPILE_CACHE_DIR:-/tmp/b12x-compile}"
  mkdir -p "$B12X_COMPILE_CACHE_DIR"
  echo "QWEN38_B12X activo rank=${VLLM_NODE_RANK} linear=b12x moe=${QWEN38_B12X_MOE:-0} wheelhouse=${B12X_WHEELHOUSE} jit=${B12X_COMPILE_CACHE_DIR}"
fi
exec taskset -c "$CPUS" vllm serve "${ARGS[@]}"
