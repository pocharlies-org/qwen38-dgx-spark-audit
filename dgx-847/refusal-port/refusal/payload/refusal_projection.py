"""Proyeccion rank-1 de refusal en runtime para Qwen3.8-Flash-Next servido con vLLM.

Hermano de `~/k8s/qwen38-27b-rank1-refusal-projection/runtime/vllm-0.27.1/`, pero
contra el modelo REAL del residente. En el FORK b12x de ursuciprian es
`vllm.models.qwen3_8_flash_next.model`; en la nightly de RadixArk el mismo modulo
se llama `vllm.models.qwen4_exp.nvidia.model` (no
`model_executor/models/qwen3_5.py`, que en este checkpoint no se instancia nunca:
el registry desvia `Qwen4ExpForConditionalGeneration` al modulo vendorizado).

QUE CAMBIA RESPECTO AL PORT DEL 27B, y por que
  1. UNA sola direccion, sin `coef` por modulo. Medido shard a shard contra
     `windowsxp811203/Qwen3.8-Flash-Next-Abliterated`: |cos| entre direcciones de
     seis clases de writer distintas 0.999996-0.999999 y lam_eff 1.4994-1.5008,
     spread 0.09% (el 27B: 0.999-1.291, spread 29%). Con spread asi, un `coef` por
     modulo no representa nada por encima del suelo de redondeo BF16. lam=1.5
     reproduce el checkpoint publicado; lam=0 es el base BIT-EXACTO.
  2. Se ablan TAMBIEN `embed_tokens` y `ple`, que el 27B no tiene. `ple.value_proj`
     es writer y se suma DIRECTO al estado multi-stream, sin pasar por ningun
     combine: si no lleva sitio propio queda intacto.
  3. El drafter MTP SI se abla. `fc_embedding`/`fc_hidden` son los analogos del
     drafter de `embed_tokens` y estan medidos como writers (lam_eff 1.4994/1.5008,
     |cos| 0.999992/0.999998). Sin ablarlos la acceptance se hunde con lam>0.

TENSOR PARALLELISM -- la parte que no es obvia, y por la que esto no es un copia-y-
pega del 27B
  El producto escalar `r_hat . y` necesita la componente de 2560 dims COMPLETA. Un
  `all_gather` a ciegas es incorrecto dos veces: porque la particion puede ser por
  columna (entonces hay que gather) y porque puede NO serlo (entonces el gather
  duplica y el producto sale multiplicado por tp_size).

  Resuelto mirando el tipo de capa, no adivinando:
    - `self_attn.o_proj` y `linear_attn.out_proj` son **RowParallelLinear** con
      `reduce_results=True` -> su salida ya esta all-reduced, cada rank tiene la
      componente completa. NO lleva gather.
    - `mlp` (FusedMoE + shared expert) idem, salida all-reduced. NO lleva gather.
    - `ple` emite al tronco multi-stream. Su `value_proj` es ColumnParallel ->
      SÍ particionado por columna.
    - `embed_tokens` es **VocabParallelEmbedding**: particiona el VOCABULARIO, no
      hidden. Cada rank tiene las filas que le tocan, con la componente de 2560
      COMPLETA. NO lleva gather, y un gather aqui seria un error de forma.

  Y hay un motivo adicional para no hacer gather "por si acaso": el parche del 27B
  trae `tensor_model_parallel_all_gather`/`reduce_scatter` en su patch_qwen, pero
  eso es el forward de v0.27.1 copiado byte a byte para preservar la forma, NO
  codigo de ablacion. Copiarlo aqui anadiria comunicacion por capa sin beneficio.

  Aun asi el helper de gather existe y esta GATEADO por asercion: si una capa
  futura cambia a `reduce_results=False` o a sequence-parallel, el build o el
  arranque tienen que morir, no proyectar sobre una particion. En este modelo
  `use_sequence_parallel_moe` esta PROHIBIDO explicitamente (model.py:153 en el
  fork, `Qwen3_8FlashNextSparseMoeBlock.__init__` lanza NotImplementedError), asi
  que el reduce_scatter no puede estar activo.

Lo demas es igual que el 27B: lam es un TENSOR en device mutado in-place (SGLang/
vLLM capturan CUDA graphs; un float de Python se hornea en la captura), lam por
peticion via `cache_salt: "refusal:<x>"`, y FAIL-CLOSED al acabar de construir el
modelo.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ENV_DIR = "SGLANG_REFUSAL_DIRECTION"   # nombre historico: se conserva del port SGLang
_ENV_DIR_VLLM = "VLLM_REFUSAL_DIRECTION"
_ENV_LAMBDA = "VLLM_REFUSAL_LAMBDA"

LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0

# El checkpoint publicado abla con 1.5. Es el punto calibrado por criterio, no un
# barrido: ver README seccion "calibracion".
LAMBDA_BAKED = 1.5

_SALT_RE = re.compile(r"refusal:\s*([-+]?\d*\.?\d+)")

# El drafter MTP tiene su propio indice de capas, que tambien empieza en 0. Sin este
# filtro reclamaria la direccion de la capa 0 del backbone.
_DRAFT_MARKERS = ("mtp", "draft", "dspark", "dflash", "nextn", "eagle")

_HIDDEN = 2560


class _State:
    """Direccion + dial global + buffer por token.

    Los tres son tensores en device mutados in-place: bajo CUDA graph el grafo
    hornea el PUNTERO, no el contenido.
    """

    __slots__ = ("dir32", "lam", "tok", "hidden", "device", "value")

    def __init__(self, direction: torch.Tensor, lam: float, device, hidden: int):
        # inference_mode(False) NO es opcional: un tensor nacido en inference mode
        # no admite mutacion in-place despues, y el dial es exactamente eso.
        with torch.inference_mode(False):
            self.dir32 = direction.to(device=device, dtype=torch.float32).contiguous()
            self.lam = torch.tensor(float(lam), device=device, dtype=torch.float32)
            self.tok = torch.full(
                (TOKEN_BUFFER_ROWS,), float(lam), device=device, dtype=torch.float32
            )
        self.hidden = int(hidden)
        self.device = device
        self.value = float(lam)

    def set_lambda(self, value: float) -> None:
        with torch.inference_mode(False):
            self.lam.fill_(float(value))
            self.tok.fill_(float(value))
        self.value = float(value)


TOKEN_BUFFER_ROWS = 131072

_STATE: Optional[_State] = None
_consumed: set = set()
# Sitios DECLARADOS al construir el modulo. Nace con el port al fork b12x
# (DGX-847, 12-10-2026): con `VLLM_USE_AOT_COMPILE=1` el grafo puede venir de un
# artefacto serializado en disco, y entonces Dynamo NO traza nada en este proceso
# -- `_consumed` se queda vacio y el guard de ejecucion mataria un servidor sano.
# `declare()` corre en `__init__`, que SIEMPRE es eager, y da el guard de
# construccion; el de ejecucion sigue siendo valido cuando el artefacto se compila
# en el arranque (cache miss).
_declared: set = set()
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("rank1-refusal: %s", msg)


def init_from_env(hidden_size: int, device=None) -> Optional[_State]:
    """Se llama en la construccion del modelo, ANTES de capturar los grafos.

    Sin variable de entorno no hay estado y todo es la identidad. Con la variable
    puesta y el fichero ilegible o de dim erronea, ABORTA: un servidor que arranca
    "casi" ablado no se distingue de uno sano mirando una respuesta.
    """
    global _STATE
    if _STATE is not None and _STATE.hidden == int(hidden_size):
        # IDEMPOTENTE: lo llaman el modelo target y la cabeza MTP, mismo proceso.
        # Un _State nuevo dejaria al grafo ya capturado del target apuntando al
        # tensor `lam` viejo: el dial dejaria de funcionar en silencio.
        return _STATE

    path = os.environ.get(_ENV_DIR_VLLM) or os.environ.get(_ENV_DIR)
    if not path:
        logger.info("rank1-refusal: sin %s, proyeccion DESACTIVADA", _ENV_DIR_VLLM)
        _STATE = None
        return None

    import numpy as np

    arr = np.load(path)
    if arr.ndim != 1 or arr.shape[0] != hidden_size:
        raise RuntimeError(
            f"rank1-refusal: {path} tiene shape {arr.shape}, se esperaba ({hidden_size},)"
        )
    vec = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    norm = float(vec.norm())
    if not (0.99 <= norm <= 1.01):
        raise RuntimeError(f"rank1-refusal: direccion no unitaria (||v||={norm:.6f})")
    vec = vec / norm

    lam = float(os.environ.get(_ENV_LAMBDA, "0"))
    if not (LAMBDA_MIN <= lam <= LAMBDA_MAX):
        raise RuntimeError(
            f"rank1-refusal: {_ENV_LAMBDA}={lam} fuera de [{LAMBDA_MIN}, {LAMBDA_MAX}]"
        )

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _STATE = _State(vec, lam, device, hidden_size)
    logger.info(
        "rank1-refusal: ACTIVA dir=%s hidden=%d lambda=%.4f device=%s",
        path, hidden_size, lam, device,
    )
    return _STATE


def is_enabled() -> bool:
    return _STATE is not None


def get_state() -> Optional[_State]:
    return _STATE


def set_lambda(value: float) -> float:
    if _STATE is None:
        raise RuntimeError("rank1-refusal: proyeccion desactivada en este servidor")
    v = float(value)
    if not (LAMBDA_MIN <= v <= LAMBDA_MAX):
        raise ValueError(f"lambda {v} fuera de [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    _STATE.set_lambda(v)
    return v


def get_lambda() -> Optional[float]:
    return None if _STATE is None else float(_STATE.lam.item())


def parse_request_lambda(extra_key) -> Optional[float]:
    """`cache_salt: "refusal:<x>"` -> lambda. None si no lo trae o no es valido.

    Por REGEX y no por prefijo: el layer OpenAI concatena cache_salt + extra_key sin
    separador. Fuera de cota se IGNORA (aviso unico) y esa peticion usa el global;
    nunca el lambda de otra.
    """
    if not extra_key or not isinstance(extra_key, str):
        return None
    m = _SALT_RE.search(extra_key)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not (LAMBDA_MIN <= v <= LAMBDA_MAX):
        _warn_once("salt_range", f"cache_salt pide lambda {v}, fuera de "
                                 f"[{LAMBDA_MIN}, {LAMBDA_MAX}]; se usa el global")
        return None
    return v


def lambda_hash_key() -> Optional[str]:
    """Clave de cache para que dos lambdas no compartan bloques de KV."""
    if _STATE is None:
        return None
    return f"{float(_STATE.value):.6f}"


# ------------------------------------------------------------------ el kernel


def _lam_tensor():
    return _STATE.tok if _STATE is not None else None


def project(y: torch.Tensor, *, key: Optional[str] = None,
            sharded: bool = False) -> torch.Tensor:
    """y (N, H) -> y - lam * r_hat (r_hat . y).

    `sharded=True` exige que la componente venga particionada por columna y hace
    all_gather ANTES del producto escalar. Esta asertivamente CARO y solo lo usan
    los sitios que de verdad lo necesitan (hoy: ninguno de los del target, ver
    docstring del modulo). Si `sharded=True` con tp_size==1, aborta: es que el
    anclaje esta mal.
    """
    st = _STATE
    if st is None:
        return y
    if key is not None:
        _consumed.add(key)

    if sharded:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp <= 1:
            raise RuntimeError(
                f"rank1-refusal: {key} marcado sharded con tp={tp}; el anclaje esta mal"
            )
        # gather por la ultima dim -> (N, tp*H_local). Es la UNICA razon de ser de
        # esta rama; cada rank queda con la componente completa.
        y_full = tensor_model_parallel_all_gather(y, -1)
        if y_full.shape[-1] != st.hidden:
            raise RuntimeError(
                f"rank1-refusal: {key} gather -> {y_full.shape[-1]}, se esperaba {st.hidden}"
            )
        return y_full - _corr(y_full, y_full.dtype).reshape(y_full.shape)

    if y.shape[-1] != st.hidden:
        # FAIL-CLOSED: una forma que no es el hidden completo en un sitio que NO se
        # declaro particionado es un anclaje mal puesto. No se proyecta a medias.
        raise RuntimeError(
            f"rank1-refusal: {key or '?'} esperaba ultimo eje {st.hidden}, "
            f"visto {tuple(y.shape)}"
        )
    return y - _corr(y, y.dtype).reshape(y.shape)


def _corr(y: torch.Tensor, out_dtype):
    """lam * r_hat (r_hat . y) sobre (N, H) ya completo."""
    st = _STATE
    flat = y.reshape(-1, st.hidden)
    lam = st.tok[: flat.shape[0]].unsqueeze(-1)     # (N,1) -- lam POR TOKEN
    coef = (flat.float() @ st.dir32).unsqueeze(-1) * lam
    return (coef * st.dir32).to(out_dtype)


def project_hc(y: torch.Tensor, hc_count: int, *, key: Optional[str] = None) -> torch.Tensor:
    """Igual sobre el tronco hiper-conectado (N, hc_count*H), por stream.

    Validado por forma, no por convencion: hc_count=4 lo exige el constructor de
    `Qwen3_8FlashNextTextConfig` (config.py:49-51) y `intermediate_size =
    hidden_size * hc_count` es el tamano del intermediate tensor del modelo.
    """
    st = _STATE
    if st is None:
        return y
    if key is not None:
        _consumed.add(key)
    expected = st.hidden * hc_count
    if y.shape[-1] != expected:
        raise RuntimeError(
            f"rank1-refusal(hc): {key or '?'} esperaba ultimo eje {expected}, "
            f"visto {tuple(y.shape)}"
        )
    shaped = y.reshape(-1, hc_count, st.hidden)
    n = shaped.shape[0] * hc_count
    lam = st.tok[:n].reshape(-1, hc_count, 1)
    coef = (shaped.float() @ st.dir32).unsqueeze(-1) * lam
    return y - (coef * st.dir32).to(y.dtype).reshape(y.shape)


def _site_kind(key: str) -> str:
    """El TIPO de sitio, no su posicion.

    Las claves del parche son:
        <prefix>.layers.<i>.attn_out.<layer_type>
        <prefix>.layers.<i>.mlp_out.<clase de mlp>
        <prefix>.layers.<i>.ple
        <prefix>.embed_tokens
        mtp.embed_tokens / mtp.fc_embedding+fc_hidden

    Se toma el primer componente que no es numerico ni 'layers'. Asi no depende de
    cuantos niveles ponga el prefijo (model.layers vs model.language_model.layers),
    que es justo lo que cambia entre checkpoints y motores.
    """
    for part in key.split("."):
        if part.isdigit() or part == "layers":
            continue
        if part in ("attn_out", "mlp_out", "ple", "embed_tokens"):
            return part.split("_")[0] if part == "embed_tokens" else part
    return ""


def verify_consumed(prefix: str, expected: int, suffix: str = "") -> None:
    """FAIL-CLOSED por PATRON, no por numero global.

    Se cuenta solo lo que empieza por `prefix`. Un numero global seria falso en
    cuanto el drafter MTP viva en el MISMO proceso que el target (que es el caso):
    sus sitios se sumarían al conteo del target y el arranque moriría sin que nada
    este mal. Contar por prefijo es lo que hace la comprobacion correcta sin
    depender del orden de construccion.
    """
    if _STATE is None:
        return
    # `suffix` es el SEGUNDO componente del sitio. Contar por startswith es
    # incorrecto aqui: "model.layers.1.ple" empieza por "model.layers", asi que
    # las 2 proyecciones por capa y los ple se mezclarian y el gate mataria un
    # servidor sano (medido: 97 frente a 96). Con sufijo explicito cada comprobacion
    # mira solo su tipo de sitio.
    got = sum(1 for k in _consumed
              if k.startswith(prefix) and _site_kind(k) == suffix)
    if got != expected:
        raise RuntimeError(
            f"rank1-refusal: {got} sitios '{prefix}*' registrados, se esperaban "
            f"{expected}. El modelo NO esta completamente ablado; no sirve."
        )
    logger.info("rank1-refusal: %d sitios '%s*' verificados", got, prefix)


def declare(key: Optional[str]) -> None:
    """Registra un sitio al CONSTRUIR el modulo (siempre eager, también bajo AOT).

    Se llama desde `__init__` con la MISMA cadena que despues usa el forward; por
    eso el parche guarda la clave en `self._rk_*` en vez de repetirla: dos copias
    de la cadena son dos oportunidades de que diverjan, y la divergencia se parece
    a un modelo sano que no abla.
    """
    if key:
        _declared.add(key)


def verify_declared(prefix: str, expected: int, suffix: str = "") -> None:
    """Guard de CONSTRUCCION: los sitios existen en el modulo que se instancio.

    Vale cuando el forward viene de un artefacto AOT cargado de disco, donde el
    guard de ejecucion no puede dispararse porque no hay trazado en el proceso.
    Lo que garantiza la ablacion en ese caso es la suma de dos cosas: esta
    comprobacion (el codigo parcheado es el que se construyo) y el checksum de
    fuente que vLLM aplica al cargar el artefacto (`_verify_source_unchanged`:
    si el .py cambio desde la compilacion, el artefacto se descarta y se
    recompila). Ninguna de las dos es opcional; de ahi que el mod NO tenga que
    tocar la clave del cache, que ya cambia solo porque el parche desplaza lineas.
    """
    if _STATE is None:
        return
    got = sum(1 for k in _declared
              if k.startswith(prefix) and _site_kind(k) == suffix)
    if got != expected:
        raise RuntimeError(
            f"rank1-refusal: {got} sitios declarados '{prefix}*'{suffix and ' ' + suffix}, "
            f"se esperaban {expected}. El arbol parcheado no es el que se construyo."
        )
    logger.info("rank1-refusal: %d sitios DECLARADOS '%s*'%s", got, prefix, suffix)


def verify_keys(*keys: str, source: str = "declared") -> None:
    """Guard por clave exacta, para los sitios que son UNOS pocos (embed, ple,
    drafter). `verify_consumed` cuenta por patron; aqui no hay patron posible."""
    if _STATE is None:
        return
    pool = _declared if source == "declared" else _consumed
    missing = sorted(k for k in keys if k not in pool)
    if missing:
        raise RuntimeError(
            f"rank1-refusal: sitios '{source}' que no estan: {missing}"
        )


# ------------------------------------------------------- lambda por peticion


def fill_target(idx_mapping_np, num_scheduled_tokens, global_lambda,
                num_tokens=None) -> None:
    """Escribe una fila de lambda por token del target, para el lote en curso.

    Corre fuera del grafo (en el input prep del model runner), y lo que el grafo
    lee es la MEMORIA del buffer. Fail-SAFE: si el layout no cuadra, el lote entero
    usa el global y se avisa una vez. Recortar o adivinar significaria aplicarle a
    una peticion el lambda de OTRA.
    """
    st = _STATE
    if st is None:
        return
    import numpy as np

    # El global lo PASA el caller (get_lambda() en el model runner). No se usa
    # st.value: el dial puede haberse movido por set_internal_state y st.value se
    # actualiza a la vez que el tensor, pero fiarse del orden de dos mutaciones es
    # exactamente el tipo de supuesto que deja un lote con el lambda viejo.
    base = float(global_lambda) if global_lambda is not None else st.value

    if idx_mapping_np is None or num_scheduled_tokens is None:
        with torch.inference_mode(False):
            st.tok.fill_(base)
        return

    req_idx = np.asarray(idx_mapping_np)
    toks = np.asarray(num_scheduled_tokens)
    n = int(toks.sum()) if toks.size else 0
    if n <= 0:
        with torch.inference_mode(False):
            st.tok.fill_(base)
        return

    # `num_tokens` es la suma REAL. Con verificacion adaptativa o especulacion,
    # sum(num_scheduled_tokens) es una COTA y puede excederla: usar la cota
    # desalinearia el offset de todas las peticiones salvo la primera, y cada una
    # recibiria el lambda de su vecina. Silencioso y peor que no tenerlo.
    if num_tokens is not None:
        n = min(n, int(num_tokens))

    per = np.full(n, base, dtype=np.float32)
    touched = False
    off = 0
    for i, cnt in enumerate(toks):
        j = int(req_idx[i]) if i < len(req_idx) else -1
        c = min(int(cnt), n - off)
        if c <= 0:
            break
        # `lam is not None` y NO `lam`: un sello valido puede ser 0.0 (base
        # explicita para ESA peticion) y `if lam` lo leeria como ausencia,
        # dejandole el lambda de otra.
        lam = _req_lambda(j)
        if lam is not None:
            per[off:off + c] = float(lam)
            touched = True
        off += c
    if not touched:
        with torch.inference_mode(False):
            st.tok.fill_(base)
        return
    if n > st.tok.shape[0]:
        _warn_once("rows", f"lote de {n} tokens excede el buffer "
                           f"({st.tok.shape[0]}); se usa el lambda GLOBAL")
        with torch.inference_mode(False):
            st.tok.fill_(base)
        return
    with torch.inference_mode(False):
        st.tok.fill_(base)
        st.tok[:n].copy_(torch.from_numpy(per), non_blocking=True)


def fill_neutral(global_lambda) -> None:
    st = _STATE
    if st is None:
        return
    with torch.inference_mode(False):
        st.tok.fill_(float(global_lambda) if global_lambda is not None else st.value)


# El buffer es UNO y lo leen ambos runners (target y draft compiten proceso).
# `fill_draft_neutral` existe para que el paso de draft no se quede con el lambda
# del ultimo lote del target, pero NO puede existir lambda por peticion en el
# drafter: el draft no lleva cache_salt propio. Se deja el global.
def fill_draft_neutral(global_lambda) -> None:
    fill_neutral(global_lambda)


_REQ_LAMBDA: dict = {}


def add_request(req_idx: int, lam) -> None:
    if lam is None:
        _REQ_LAMBDA.pop(int(req_idx), None)
    else:
        _REQ_LAMBDA[int(req_idx)] = float(lam)


def remove_request(req_idx: int) -> None:
    _REQ_LAMBDA.pop(int(req_idx), None)


def _req_lambda(req_idx: int):
    return _REQ_LAMBDA.get(int(req_idx)) if req_idx >= 0 else None


__all__ = [
    "LAMBDA_BAKED",
    "LAMBDA_MAX",
    "LAMBDA_MIN",
    "add_request",
    "declare",
    "fill_draft_neutral",
    "fill_neutral",
    "fill_target",
    "get_lambda",
    "get_state",
    "init_from_env",
    "is_enabled",
    "lambda_hash_key",
    "parse_request_lambda",
    "project",
    "project_hc",
    "remove_request",
    "set_lambda",
    "verify_consumed",
]
