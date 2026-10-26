#!/usr/bin/env bash
set -euo pipefail

# 10-10-2026 (DGX-781): TP=3 por el triangulo CX7. Cada Spark habla con los otros dos por SUS DOS HCA, uno por vecino
# (NCCL_IB_HCA, la misma lista en los tres rangos; mapa de cables en dgx-infra/scripts/roce-link/ring.conf). Antes de cargar
# nada se comprueba cada HCA de esa lista: puerto 1 ACTIVE, indice del GID RoCE v2 IPv4 y MTU de su netdev. Los dos tienen
# que dar el MISMO indice y la MISMA MTU. Si no, exit 7 sin lanzar vllm: con un enlace a medias el TP=3 no falla al
# arrancar, se cuelga en el primer colectivo. Antes (TP=2) solo se miraba el estado de rocep1s0f0.
#
# 10-10-2026 (DGX-817): con la libnccl parcheada de nccl-ring (la monta el overlay de la ventana, que pone QWEN_NCCL_RING_HOST) el grafo
# y NCCL_IB_HCA salen del generador de esa imagen, como en la sonda v3c que paso (DGX-810, DGX-815, 23 GB/s con 256 MiB): las cuatro
# funciones RDMA del nodo, un canal recibe por la puerta que mira al rango anterior y envia por la del siguiente. QWEN_NCCL_RING_HOST es el
# host de ring.conf de ESTE rango (dgx1, dgx2 o dgx3, SONDA_NODO en los Jobs de la sonda): lo dice el manifiesto, no se adivina. El generador
# corre ANTES del preflight, que mira la lista que NCCL va a usar. Sin QWEN_NCCL_RING_HOST no hace nada y NCCL_IB_HCA es el del manifiesto.
if [ -n "${QWEN_NCCL_RING_HOST:-}" ]; then
  RING_OUT=$(python3 -I /opt/nccl-ring/nccl_ring_graph.py --host "$QWEN_NCCL_RING_HOST" --channels 2 --out /tmp/nccl-graph.xml) || {
    echo "QWEN_TP3_PREFLIGHT ABORTA: el generador del grafo falla con host='$QWEN_NCCL_RING_HOST'" >&2; exit 7; }
  while IFS= read -r ln; do export "${ln%%=*}=${ln#*=}"; done <<< "$RING_OUT"
  echo "QWEN_NCCL_RING host=$QWEN_NCCL_RING_HOST NCCL_IB_HCA=${NCCL_IB_HCA:?el generador no dio NCCL_IB_HCA} NCCL_GRAPH_FILE=${NCCL_GRAPH_FILE:?el generador no dio NCCL_GRAPH_FILE}"
  cat "$NCCL_GRAPH_FILE"
fi
IB_SYSFS="${IB_SYSFS:-/sys/class/infiniband}"
IFS=, read -r -a HCAS <<< "${NCCL_IB_HCA#=}"
PRE_GID="" PRE_MTU=""
MIN_HCAS=2  # con el anillo del generador, las cuatro funciones de ring.conf
if [ -n "${QWEN_NCCL_RING_HOST:-}" ]; then MIN_HCAS=4; fi
if [ "${#HCAS[@]}" -lt "$MIN_HCAS" ]; then
  echo "QWEN_TP3_PREFLIGHT ABORTA: NCCL_IB_HCA='${NCCL_IB_HCA:-}' trae ${#HCAS[@]} HCA y hacen falta ${MIN_HCAS}" >&2; exit 7
fi
for hca in "${HCAS[@]}"; do
  port="$IB_SYSFS/$hca/ports/1"
  est=$(cat "$port/state" 2>/dev/null || echo desconocido)
  case "$est" in
    *ACTIVE*) ;;
    *) echo "QWEN_TP3_PREFLIGHT ABORTA: $hca puerto 1 no esta ACTIVE (state='$est'). Cable DAC?" >&2; exit 7 ;;
  esac
  gid=""
  for i in $(ls "$port/gid_attrs/types" 2>/dev/null | sort -n || true); do
    [ "$(cat "$port/gid_attrs/types/$i" 2>/dev/null || true)" = "RoCE v2" ] || continue
    g=$(cat "$port/gids/$i" 2>/dev/null || true)
    case "$g" in 0000:0000:0000:0000:0000:ffff:*) gid=$i; break ;; esac
  done
  mtu=$(cat "$IB_SYSFS/$hca"/device/net/*/mtu 2>/dev/null || true)
  case "$gid:$mtu" in
    :*|*:|*:*[!0-9]*) echo "QWEN_TP3_PREFLIGHT ABORTA: $hca sin GID RoCE v2 IPv4 o sin MTU (gid='$gid' mtu='$mtu')" >&2; exit 7 ;;
  esac
  PRE_GID="${PRE_GID:-$gid}" PRE_MTU="${PRE_MTU:-$mtu}"
  if [ "$gid" != "$PRE_GID" ] || [ "$mtu" != "$PRE_MTU" ]; then
    echo "QWEN_TP3_PREFLIGHT ABORTA: $hca gid=$gid mtu=$mtu frente a gid=$PRE_GID mtu=$PRE_MTU del primero" >&2; exit 7
  fi
done
echo "QWEN_TP3_PREFLIGHT ok rank=${VLLM_NODE_RANK} hcas=${NCCL_IB_HCA#=} gid=${PRE_GID} mtu=${PRE_MTU}"

# Pinning de CPU. La receta de origen usa `taskset -c 5-9,15-19`: 10 cores
# concretos. En decode el cuello es el lanzamiento de kernels desde CPU, no
# la GPU, asi que esto no es cosmetico.
CPUS="${VLLM_CPUSET:-5-9,15-19}"

ARGS=(
  "$MODEL_PATH"
  --revision "$MODEL_REVISION"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 8000
  # 10-10-2026 (DGX-781): TP=3 + EP=3, un rango por Spark (head gx10-ec3d, worker nvidia-dgx, worker-dgx3 dgx3).
  # ROLLBACK: pin 63dd55f de la Application `ai` (TP=2), en su PR de k8s-gitops-pocharlies.
  --tensor-parallel-size 3
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
  --nnodes 3
  --node-rank "$VLLM_NODE_RANK"
  --master-addr "$VLLM_MASTER_ADDR"
  --master-port 29501
  --trust-remote-code
  # DGX-781: la torre visual (16 cabezas, intermediate 4304) no se parte en 3; cada rango la corre entera
  # (DGX-780 C8: Qwen3VL declara supports_encoder_tp_data, .company/evidence/DGX-780-tp3-parches.md).
  --mm-encoder-tp-mode data
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
# 27-09-2026 (Dani, ablacion de Jordi): estado SSM de Mamba/GDN en BF16. El config del
# modelo declara mamba_ssm_dtype=float32 y vLLM avisa del override (models/config.py:820).
# Jordi, misma imagen 96b234: prefill incremental tras 64k +50/+53 % y TTFR -37/-38 % a
# c4/c8; decode mixto (c4 -7 %, c8 +6 %, corto/c4 -11 %). Calidad: longctx_eval.py
# (needle 32k-190k + continuacion greedy) frente a dos pasadas FP32. Vacio = auto (FP32).
if [ -n "${MAMBA_SSM_CACHE_DTYPE:-}" ]; then
  ARGS+=( --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE" )
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
  # 28-09-2026: config de atencion (p. ej. indexer_kv_dtype de la QSA). Vacio = por defecto.
  ${VLLM_ATTENTION_CONFIG:+--attention-config "$VLLM_ATTENTION_CONFIG"}
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
echo "QWEN_VLLM_EFFECTIVE rank=${VLLM_NODE_RANK} kv=${KV_CACHE_DTYPE} seqs=${MAX_NUM_SEQS} memutil=${GPU_MEM_UTIL} kvbytes=${KV_CACHE_MEMORY_BYTES:-0} headless=${HEADLESS_ON} v2runner=${VLLM_USE_V2_MODEL_RUNNER:-default} long_prefill_threshold=${LONG_PREFILL_TOKEN_THRESHOLD:-0} mamba_ssm=${MAMBA_SSM_CACHE_DTYPE:-auto} ple=${QWEN_PLE_MMAP_DIR:-gpu} scheduling_policy=$([ "${VLLM_NODE_RANK}" = "0" ] && echo "${SCHEDULING_POLICY:-fcfs}" || echo n/a)"
# 25-09-2026: backport de vllm#48329 (arregla vllm#58692). Con Dynamic SD
# (num_speculative_tokens_per_batch_size) el gestor de cudagraphs del
# DECODE del drafter MTP (decode_query_len=1) aplicaba el calendario del
# modelo target: 1 - k_max = -3 -> query lens {-2,-1,0,1} -> round_up(x, 0)
# -> ZeroDivisionError en _init_candidates. El guard limita el calendario al
# gestor cuyo decode_query_len cubre la verificacion. Idempotente; si el
# patron no esta (imagen nueva) avisa y sigue sin tocar nada.
python3 - <<'PYEOF'
import pathlib, vllm
f = pathlib.Path(vllm.__file__).parent / 'v1/worker/gpu/cudagraph_utils.py'
s = f.read_text()
old = ("            and speculative_config.uses_dynamic_speculative_decoding()\n"
       "        ):\n")
new = ("            and speculative_config.uses_dynamic_speculative_decoding()\n"
       "            and self.decode_query_len > self.vllm_config.num_speculative_tokens\n"
       "        ):\n")
if new in s:
    print('QWEN_PATCH_58692 ya aplicado')
elif s.count(old) == 1:
    f.write_text(s.replace(old, new))
    print('QWEN_PATCH_58692 aplicado')
else:
    print('QWEN_PATCH_58692 AVISO: patron no encontrado, sin cambios')
PYEOF
# 09-10-2026 (DGX-773): la tabla PLE (23,84 GiB por rango, la mitad de las 320.001.536 filas x 160 B del checkpoint) sale de la GPU y
# se sirve desde el NVMe de ESTE nodo: cada rango TP lee solo sus filas de su fichero local (hostPath), sin NFS ni el otro nodo, y
# copia a la GPU solo las de cada paso (ple_mmap.py, codigo propio; el enfoque es el del lab de dgx3, sin el codigo AGPL de MiaAI-Lab,
# que ademas no casa con esta imagen: la nightly 2a02f6ef ya no trae v1/ple_offload ni el offload nativo con nnodes=2 que lo rechazaba).
# Corre en los DOS rangos (head y worker ejecutan este mismo script) y, a diferencia de los parches de arriba (que avisan y siguen), CIERRA en fallo: sin la tabla construida por
# el initContainer `ple-table` (marca ready.json) sale con 71 sin lanzar vllm; un ancla que no casa una sola vez, un fichero a medias o
# una marca que no es de este rango = el pod no arranca (con la bandera puesta, seguir con la tabla en GPU duplicaria la memoria que
# la cuenta de KV_CACHE_MEMORY_BYTES ya dio por liberada). Sin QWEN_PLE_MMAP_DIR no hace nada. Anclas probadas contra los ficheros de la
# imagen en tools/test_qwen38_flashinfer_contract.py. Antes de lanzar vllm suelta de la cache de paginas lo que quedara de la tabla de
# un arranque anterior: en GB10 CUDA no cuenta la cache como libre y la puerta de GPU_MEM_UTIL del head tiene ~4,8 GiB de margen.
if [ -n "${QWEN_PLE_MMAP_DIR:-}" ]; then
  if [ ! -f "$QWEN_PLE_MMAP_DIR/ready.json" ] || [ ! -f "$QWEN_PLE_MMAP_DIR/table.u8" ]; then
    echo "QWEN_PLE_MMAP ABORTA: faltan $QWEN_PLE_MMAP_DIR/ready.json o table.u8 (los deja el initContainer ple-table solo si termina bien)" >&2
    exit 71
  fi
  python3 - <<'PYEOF'
import os, pathlib, shutil, sys, vllm
ROOT = pathlib.Path(vllm.__file__).parent
NV = ROOT / 'models/qwen4_exp/nvidia'
SRC = pathlib.Path(os.environ.get('QWEN_PLE_SRC', '/opt/qwen38fn-ple')) / 'ple_mmap.py'
NGRAM_OLD = '        self.ngram_embedding = embedding_cls(\n'
NGRAM_NEW = (
    '        # QWEN_PLE_MMAP (DGX-773): con QWEN_PLE_MMAP_DIR la tabla PLE es la del NVMe local (ple_mmap.py), no la de la GPU.\n'
    '        from .ple_mmap import select_embedding_cls\n'
    '\n'
    '        embedding_cls = select_embedding_cls(embedding_cls)\n'
) + NGRAM_OLD
STATE_OLD = (
    '            ngram_context=self._prepare_ngram_context(input_batch, req_states),\n'
    '        )\n'
    '        return model_inputs\n'
)
STATE_NEW = (
    '            ngram_context=self._prepare_ngram_context(input_batch, req_states),\n'
    '        )\n'
    '        # QWEN_PLE_MMAP (DGX-773): fuera del grafo, las filas PLE de este rango del NVMe a la GPU antes del paso.\n'
    '        from .ple_mmap import fill_ple_rows\n'
    '\n'
    '        fill_ple_rows(self.model, input_batch, model_inputs)\n'
    '        return model_inputs\n'
)
EDITS = {'ngram_embedding.py': (NGRAM_OLD, NGRAM_NEW), 'model_state.py': (STATE_OLD, STATE_NEW)}
if not SRC.is_file():
    sys.exit('QWEN_PLE_MMAP ABORTA: falta ' + str(SRC))
compile(SRC.read_text(), str(SRC), 'exec')
texts = {name: (NV / name).read_text() for name in EDITS}
done = [name for name, text in texts.items() if 'QWEN_PLE_MMAP' in text]
if done and len(done) != len(EDITS):
    sys.exit('QWEN_PLE_MMAP ABORTA: parche a medias, solo en ' + ', '.join(done))
if not done:
    for name, (old, new) in EDITS.items():
        if texts[name].count(old) != 1:
            sys.exit('QWEN_PLE_MMAP ABORTA: ancla no unica o ausente en ' + name + ' (count=' + str(texts[name].count(old)) + '): ' + repr(old[:80]))
        texts[name] = texts[name].replace(old, new)
        compile(texts[name], name, 'exec')
    for name, text in texts.items():
        (NV / name).write_text(text)
shutil.copyfile(SRC, NV / 'ple_mmap.py')
table = pathlib.Path(os.environ['QWEN_PLE_MMAP_DIR']) / 'table.u8'
fd = os.open(table, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
print('QWEN_PLE_MMAP ' + ('ya aplicado' if done else 'aplicado') + ' en ' + ', '.join(EDITS) + '; tabla', table, table.stat().st_size, 'bytes')
PYEOF
fi
# 27-09-2026: top-k determinista de la QSA (jschmied, vllm#55122; bug vllm#51782). En GB10 el
# persistent_topk stock cambia el ORDEN y a veces el CONJUNTO de bloques elegidos entre
# ejecuciones (la rama cooperativa excluye la familia 120, asi que siempre cae aqui).
# Se compila con el nvcc de la imagen ANTES de cargar pesos (~1 min, memoria libre), se cachea
# en /tmp (hostPath) por hash de fuentes + version de torch, y test_det.py corre una vez por
# build. Si el build o el test fallan -> kernel stock, como hasta hoy. Fuentes y procedencia:
# k8s/qwen38-flash-next-kernel-det/. Apagar: VLLM_QSA_DET_TOPK=0.
if [ "${VLLM_QSA_DET_TOPK:-0}" = "1" ]; then
  KDET_SRC=/opt/qwen38fn-kdet
  if (cd "$KDET_SRC" && sha256sum -c --quiet SHA256SUMS); then
    KDET_ID=$( (cat "$KDET_SRC"/SHA256SUMS; python3 -c 'import torch;print(torch.__version__, torch.version.cuda)') | sha256sum | cut -c1-16)
    KDET_DIR=/tmp/qwen38fn-kdet-$KDET_ID
    if [ ! -f "$KDET_DIR/OK" ]; then
      rm -rf "$KDET_DIR"; mkdir -p "$KDET_DIR/src"
      cp -L "$KDET_SRC"/*.cu "$KDET_SRC"/*.cpp "$KDET_SRC"/*.cuh "$KDET_SRC"/*.h "$KDET_SRC"/*.py "$KDET_DIR/src/"
      echo "QWEN_KDET compilando en $KDET_DIR"
      if (cd "$KDET_DIR/src" && DET_BUILD_DIR="$KDET_DIR/build" DET_ARCH=121a timeout 900 python3 build_det.py > "$KDET_DIR/build.log" 2>&1) \
         && (cd "$KDET_DIR/src" && timeout 900 python3 test_det.py "$KDET_DIR/build/_C_det.so" > "$KDET_DIR/test.log" 2>&1) \
         && grep -q "^FAILS: 0" "$KDET_DIR/test.log"; then
        touch "$KDET_DIR/OK"
      else
        echo "QWEN_KDET AVISO: build o test fallido, se queda el kernel stock"; tail -5 "$KDET_DIR/build.log" "$KDET_DIR/test.log" 2>/dev/null || true
      fi
    fi
    if [ -f "$KDET_DIR/OK" ]; then
      export VLLM_QSA_DET_LIB="$KDET_DIR/build/_C_det.so"
      echo "QWEN_KDET listo: $VLLM_QSA_DET_LIB ($(grep -c '^OK' "$KDET_DIR/test.log") casos OK)"
      python3 - <<'PYEOF'
import pathlib, vllm
f = pathlib.Path(vllm.__file__).parent / 'models/qwen4_exp/nvidia/ops/qsa_indexer.py'
s = f.read_text()
old = ("    topk_op = (\n"
       "        torch.ops._C.cooperative_topk\n"
       "        if use_cooperative_topk\n"
       "        else torch.ops._C.persistent_topk\n"
       "    )\n")
new = ("    # QWEN_KDET (vllm#55122): top-k determinista, cargado por launch.sh\n"
       "    import os as _os\n"
       "    if _os.environ.get('VLLM_QSA_DET_TOPK') == '1' and _os.environ.get('VLLM_QSA_DET_LIB'):\n"
       "        if not getattr(_topk, '_kdet_loaded', False):\n"
       "            torch.ops.load_library(_os.environ['VLLM_QSA_DET_LIB'])\n"
       "            _topk._kdet_loaded = True\n"
       "            print('QWEN_KDET activo en _topk', flush=True)\n"
       "        topk_op = torch.ops._C_det.persistent_topk\n"
       "    else:\n"
       "        topk_op = (\n"
       "            torch.ops._C.cooperative_topk\n"
       "            if use_cooperative_topk\n"
       "            else torch.ops._C.persistent_topk\n"
       "        )\n")
if 'QWEN_KDET' in s:
    print('QWEN_KDET ya cableado')
elif s.count(old) == 1 and 'def _topk(' in s:
    f.write_text(s.replace(old, new))
    print('QWEN_KDET cableado en', f)
else:
    print('QWEN_KDET AVISO: patron no encontrado, kernel stock')
PYEOF
    fi
  else
    echo "QWEN_KDET AVISO: SHA256SUMS no cuadra, kernel stock"
  fi
fi
# 09-10-2026 (DGX-780): parches de arranque para TP=3. Los tres leen QWEN_TP_SIZE (el
# --tensor-parallel-size de ARGS). Con TP=2 son INERTES: no abren el fichero de la imagen.
# Con tp % 3 == 0 son FAIL-CLOSED: si el patron no esta (imagen nueva), exit 1 y el pod no
# arranca. Idempotentes por marcador. Van despues de install.py (ursucipian), que exige los
# ficheros base sin tocar y ya ha parcheado mtp.py. Tests sin GPU sobre copias congeladas de la
# imagen 96b234af: tools/test_qwen38fn_fi_ep_uneven.py y tools/test_qwen38fn_tp3_patches.py.
export QWEN_TP_SIZE
QWEN_TP_SIZE=$(printf '%s\n' "${ARGS[@]}" | awk 'take { print; take = 0 } $0 == "--tensor-parallel-size" { take = 1 }')
# 08-10-2026 (DGX-672), portado en DGX-780: FlashInfer CUTLASS con EP desigual (512 expertos /
# EP=3 = 171/171/170, vLLM los reparte desde 0/171/342). El binding de FlashInfer solo recibe
# ep_size/ep_rank y supone reparto igual, start = ep_rank * num_experts_on_rank
# (flashinfer_cutlass_fused_moe_binding.cu:397-399): el rango 2 empezaria en 340 y no en
# 342, y el kernel descarta los ids fuera de [start, end) (cutlass_fused_moe_kernels.cuh
# :438-441, :1970-1973, :2039-2041). Antes de llamar al kernel, los ids globales se pasan a
# la numeracion que el binding supone: slot local + start; un experto que no es de este
# rango (-1) cae en start-1 y el kernel lo descarta. ep_size/ep_rank no cambian. Sin
# expert_map falla en voz alta en el primer forward. Sin sincronizacion GPU, compatible con
# la captura de CUDA graph. EPLB y expertos redundantes no estan activos; no se ha probado
# con ellos. Se aplica con tp % 3 == 0 o E % ep != 0 (E del config.json del modelo); con
# E % ep == 0 y TP=2, inerte. En cada rango del arranque TP=3 hay que ver
# "QWEN_FI_EP_UNEVEN (aplicado|ya aplicado)" y, en el primer forward, "QWEN_FI_EP_UNEVEN activo".
python3 - <<'PYEOF'
import json, os, pathlib, sys
tp = int(os.environ['QWEN_TP_SIZE'])
try:
    cfg = json.loads((pathlib.Path(os.environ['MODEL_PATH']) / 'config.json').read_text())
    experts = int(cfg.get('text_config', cfg)['num_experts'])
except (KeyError, OSError, ValueError):
    experts = None
if tp % 3 and (experts is None or experts % tp == 0):
    print(f'QWEN_FI_EP_UNEVEN inerte (E={experts}, ep={tp})')
    sys.exit(0)
import vllm
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
    print(f'QWEN_FI_EP_UNEVEN ABORTA: patron no encontrado con E={experts}, ep={tp}', file=sys.stderr)
    sys.exit(1)
PYEOF
# DGX-780: las dos FC del MTP (fc_embedding, fc_hidden) son ColumnParallelLinear de 2560 a 2560
# con gather_output, y 2560 no se parte en 3. Con tp % 3 == 0 pasan a ReplicatedLinear: cada
# rango hace la FC entera (2 x 2560 x 2560 BF16, 25 MiB por rango) y el resultado es el mismo
# que el gather. El checkpoint trae los pesos enteros, que es lo que carga ReplicatedLinear.
python3 - <<'PYEOF'
import os, pathlib, sys
tp = int(os.environ['QWEN_TP_SIZE'])
if tp % 3:
    print(f'QWEN_TP3_MTP_FC_REPLICATED inerte (tp={tp})')
    sys.exit(0)
import vllm
f = pathlib.Path(vllm.__file__).parent / 'models/qwen4_exp/nvidia/mtp.py'
s = f.read_text()
pairs = [("from vllm.model_executor.layers.linear import ColumnParallelLinear\n",
          "from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear  # QWEN_TP3_MTP_FC_REPLICATED\n")]
for name in ('fc_embedding', 'fc_hidden'):
    pairs.append((f"            self.{name} = ColumnParallelLinear(\n"
                  "                self.hidden_size,\n"
                  "                self.hidden_size,\n"
                  "                gather_output=True,\n",
                  "            # QWEN_TP3_MTP_FC_REPLICATED (DGX-780): 2560 no se parte en 3\n"
                  f"            self.{name} = ReplicatedLinear(\n"
                  "                self.hidden_size,\n"
                  "                self.hidden_size,\n"))
if 'QWEN_TP3_MTP_FC_REPLICATED' in s:
    print('QWEN_TP3_MTP_FC_REPLICATED ya aplicado')
elif all(s.count(old) == 1 for old, _ in pairs):
    for old, new in pairs:
        s = s.replace(old, new)
    f.write_text(s)
    print('QWEN_TP3_MTP_FC_REPLICATED aplicado')
else:
    print(f'QWEN_TP3_MTP_FC_REPLICATED ABORTA: patron no encontrado con tp={tp}', file=sys.stderr)
    sys.exit(1)
PYEOF
# DGX-780: el checkpoint TP=3 lleva vocab_size 248448 (3 x 82816) con las 128 filas nuevas de
# embed_tokens/lm_head a cero; su config guarda el vocabulario de TP=2 en tp3_orig_vocab_size
# (248320). Esos logits valdrian 0, no -inf, y podrian salir en el muestreo. Con tp % 3 == 0 se
# ponen a -inf en LogitsProcessor._apply_head, por donde pasan las tres rutas (_get_logits:
# muestreo y prompt_logprobs; get_top_tokens y get_top_k_tokens: el borrador MTP), antes del
# gather. Sin tp3_orig_vocab_size menor que vocab_size en el config, ABORTA: con TP=3 seria el
# checkpoint equivocado.
python3 - <<'PYEOF'
import json, os, pathlib, sys
tp = int(os.environ['QWEN_TP_SIZE'])
if tp % 3:
    print(f'QWEN_TP3_VOCAB_MASK inerte (tp={tp})')
    sys.exit(0)
cfg = json.loads((pathlib.Path(os.environ['MODEL_PATH']) / 'config.json').read_text())
tc = cfg.get('text_config', cfg)
real, vocab = int(tc.get('tp3_orig_vocab_size', 0)), int(tc['vocab_size'])
if not 0 < real < vocab:
    print(f'QWEN_TP3_VOCAB_MASK ABORTA: tp3_orig_vocab_size={real} y vocab_size={vocab} con tp={tp}', file=sys.stderr)
    sys.exit(1)
import vllm
f = pathlib.Path(vllm.__file__).parent / 'model_executor/layers/logits_processor.py'
s = f.read_text()
old = "    def _apply_head(\n"
new = ("    def _apply_head(self, lm_head, hidden_states, embedding_bias):\n"
       f"        # QWEN_TP3_VOCAB_MASK (DGX-780): los ids >= {real} son relleno del checkpoint TP=3\n"
       "        logits = self._qwen_tp3_apply_head(lm_head, hidden_states, embedding_bias)\n"
       f"        cut = {real} - lm_head.shard_indices.org_vocab_start_index\n"
       "        if cut < logits.shape[-1]:\n"
       "            logits[..., max(cut, 0):] = -float('inf')\n"
       "        return logits\n"
       "\n"
       "    def _qwen_tp3_apply_head(\n")
if 'QWEN_TP3_VOCAB_MASK' in s:
    print('QWEN_TP3_VOCAB_MASK ya aplicado')
elif s.count(old) == 1:
    f.write_text(s.replace(old, new))
    print(f'QWEN_TP3_VOCAB_MASK aplicado (-inf en ids >= {real} de {vocab})')
else:
    print(f'QWEN_TP3_VOCAB_MASK ABORTA: patron no encontrado con tp={tp}', file=sys.stderr)
    sys.exit(1)
PYEOF
# DGX-780 (C7b): con TP=3 la vista sube make_ngram_vocab_size_divisible_by a 384 para que la PLE
# (320.001.536 filas) se parta en 3, y Qwen4ExpNGramEmbedding.load_weights (ngram_embedding.py
# :908-921 de la imagen) exige entonces shards de 2.500.014 filas; el checkpoint trae 2.500.012.
# Con la PLE en el NVMe (QWEN_PLE_MMAP_DIR, bloque QWEN_PLE_MMAP de arriba) esas filas no se
# cargan: el weight_loader de ple_mmap.py las descarta. Con tp % 3 == 0 los shards se saltan
# el chequeo de forma y el weight_loader, y conservan loaded.add("ngram_embedding.weight"), que
# vLLM pide para dar el parametro por cargado; la escala FP8 global se carga igual. Sin la
# eleccion de clase de QWEN_PLE_MMAP en el fichero, ABORTA: la tabla quedaria en la GPU vacia.
QWEN_TP3_PLE_DIR="${QWEN_PLE_MMAP_DIR:-}" python3 - <<'PYEOF'
import os, pathlib, sys
tp, ple = int(os.environ['QWEN_TP_SIZE']), os.environ.get('QWEN_TP3_PLE_DIR', '')
if tp % 3 or not ple:
    print(f"QWEN_TP3_PLE_SHARD_SKIP inerte (tp={tp}, ple={ple or 'gpu'})")
    sys.exit(0)
import vllm
f = pathlib.Path(vllm.__file__).parent / 'models/qwen4_exp/nvidia/ngram_embedding.py'
s = f.read_text()
old = ("                embedding = self.ngram_embedding\n"
       "                shard_size = (\n")
new = ("                # QWEN_TP3_PLE_SHARD_SKIP (DGX-780): las filas salen del NVMe (ple_mmap.py)\n"
       "                loaded.add(\"ngram_embedding.weight\")\n"
       "                continue\n"
       + old)
if 'QWEN_TP3_PLE_SHARD_SKIP' in s:
    print('QWEN_TP3_PLE_SHARD_SKIP ya aplicado')
elif s.count(old) == 1 and 'from .ple_mmap import select_embedding_cls' in s:
    f.write_text(s.replace(old, new))
    print('QWEN_TP3_PLE_SHARD_SKIP aplicado')
else:
    print(f'QWEN_TP3_PLE_SHARD_SKIP ABORTA: patron no encontrado con tp={tp} (o sin la clase de ple_mmap)', file=sys.stderr)
    sys.exit(1)
PYEOF
exec taskset -c "$CPUS" vllm serve "${ARGS[@]}"
