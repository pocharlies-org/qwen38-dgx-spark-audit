"""Sonda de refusal por token (FASE B del detector): el veredicto sale del modelo.

QUE MIDE. El coseno entre el residual (la suma de los 4 streams de las
hiper-conexiones, justo antes del MLP de la capa) y la direccion de refusal que
el dial rank-1 ya carga (`refusal_projection._STATE.dir32`). Es la lectura de Arditi et al. 2024: el
rechazo esta mediado por UNA direccion, y su proyeccion en el ultimo token del
prompt predice si el modelo va a negarse ANTES de escribir nada. La heuristica de
texto (`refusal_text` en LiteLLM, FASE A) solo ve las palabras; esto ve la
intencion.

QUE NO ES. No es un veredicto hasta que hay calibracion: un coseno sin umbral es
una distribucion. `flag` solo existe si `VLLM_REFUSAL_PROBE_CAL` apunta a un JSON
valido; sin el, la respuesta lleva `score` y `flag: null`, y el panel lo pinta
como "sonda presente, sin calibrar" (ver dgx-infra `api/refusal_rates.py`).

EN LAS RUTAS ABLADAS (lambda > 0) mide "se habria negado", no "se nego": la sonda
lee el residual, que ya lleva restado lo que las capas anteriores escribian en
la direccion, asi que el valor BAJA con lambda. El sello de LiteLLM guarda
`refusal_lambda` junto al score precisamente para no mezclar las dos lecturas.

RUTA DEL DATO, de punta a punta:
  1. model.py   `probe(hidden_states, layer_idx, prefix, hc)` -> buf[token, capa] (GPU,
                 dentro del CUDA graph de decode: solo escribe en un buffer fijo)
  2. runner     `step_stats(...)` en sample_tokens -> (num_reqs, 3+K) en GPU y copia
                 asincrona a CPU en el main stream, ANTES del AsyncOutput: el
                 copy_stream espera al main stream, asi que el copy_event cubre
                 tambien esta copia. Cero sincronizaciones nuevas.
  3. scheduler  `accumulate(request, rp, idx)` por paso; `finalize(request)` al
                 terminar -> EngineCoreOutput.refusal_probe
  4. API        RequestOutput.refusal_probe -> `provider_specific_fields` en el
                 chunk final (stream) o clave extra del choice (no-stream). Es la
                 forma que LiteLLM ya propaga sin tocarle nada.

FAIL-SAFE, NO FAIL-CLOSED. Al reves que el dial: esto es telemetria. Una forma
inesperada desactiva la sonda con un aviso y el servidor sigue sirviendo igual.
Un servidor caido por una metrica es peor que una metrica ausente.

COSTE. Por capa sondeada y token: un producto escalar de 2560 y una norma, y una
escritura de un float. Con K=5 capas sobre 48 es del orden de 1e4 FLOPs por token
frente a los ~1e9 del forward. Por paso: ~8 kernels minusculos sobre (tokens, K)
y una copia D2H de num_reqs*(3+K) floats que viaja con la de los tokens.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_ENV_ON = "VLLM_REFUSAL_PROBE"
_ENV_LAYERS = "VLLM_REFUSAL_PROBE_LAYERS"
_ENV_CAL = "VLLM_REFUSAL_PROBE_CAL"
_ENV_FEATURE = "VLLM_REFUSAL_PROBE_FEATURE"

# 48 capas en Qwen3.8-Flash-Next. La literatura situa la direccion de refusal en
# el tercio central; la calibracion decide cual pesa, por eso viajan todas.
DEFAULT_LAYERS = (16, 20, 24, 28, 32)
DEFAULT_FEATURE = "prompt_last_mean"
# Ventana de generacion: el rechazo se decide en los primeros tokens. Pasada la
# ventana no se acumula mas (una respuesta de 8k tokens no diluye la media).
GEN_WINDOW = 64
BUFFER_ROWS = 131072
# El drafter MTP instancia Qwen4ExpDecoderLayer con su propio indice de capas,
# que tambien empieza en 0: sin este filtro su capa 16 pisaria la del target.
_DRAFT_MARKERS = ("mtp", "draft", "dspark", "dflash", "nextn", "eagle")

FEATURES = (
    "prompt_last_mean", "prompt_max", "prompt_mean", "gen_max", "gen_mean",
)


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def parse_layers(raw: Optional[str]) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_LAYERS
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        return DEFAULT_LAYERS
    return tuple(dict.fromkeys(out))


class _Probe:
    __slots__ = ("dir", "buf", "col_of", "layers", "hidden", "device")

    def __init__(self, direction: torch.Tensor, layers: tuple[int, ...]):
        self.dir = direction
        self.hidden = int(direction.shape[0])
        self.device = direction.device
        self.layers = layers
        self.col_of = {layer: i for i, layer in enumerate(layers)}
        with torch.inference_mode(False):
            self.buf = torch.zeros(
                (BUFFER_ROWS, len(layers)), device=self.device, dtype=torch.float32
            )


_P: Optional[_Probe] = None
_DISABLED = False
_draft_cache: dict[str, bool] = {}
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("refusal-probe: %s", msg)


def disable(reason: str) -> None:
    global _P, _DISABLED
    _warn_once("disabled", f"DESACTIVADA: {reason}")
    _P = None
    _DISABLED = True


def enabled_by_env() -> bool:
    return _truthy(os.environ.get(_ENV_ON))


def _try_init() -> Optional[_Probe]:
    """Perezoso pero antes de la captura: la primera llamada es el profile run
    eager, que corre antes de capturar los grafos. El buffer ya existe cuando el
    grafo hornea su puntero."""
    global _P, _DISABLED
    if _P is not None or _DISABLED:
        return _P
    if not enabled_by_env():
        _DISABLED = True
        return None
    try:
        from vllm import refusal_projection as _rp
    except Exception as e:  # pragma: no cover - solo fuera del site
        disable(f"sin refusal_projection ({e})")
        return None
    st = _rp.get_state()
    if st is None:
        disable("el dial no tiene direccion cargada (VLLM_REFUSAL_DIRECTION)")
        return None
    try:
        layers = parse_layers(os.environ.get(_ENV_LAYERS))
    except ValueError as e:
        disable(f"{_ENV_LAYERS} ilegible: {e}")
        return None
    _P = _Probe(st.dir32, layers)
    logger.info("refusal-probe: ACTIVA capas=%s hidden=%d device=%s",
                list(layers), _P.hidden, _P.device)
    return _P


def init_eager() -> None:
    """Desde la construccion del modelo, justo despues de cargar el dial."""
    try:
        _try_init()
    except Exception as e:
        disable(f"init: {e}")


def init_with_direction(direction: torch.Tensor, layers=DEFAULT_LAYERS) -> _Probe:
    """Para tests: sin pasar por el entorno ni por el dial."""
    global _P, _DISABLED
    _P = _Probe(direction.float().contiguous(), tuple(layers))
    _DISABLED = False
    return _P


def reset() -> None:
    global _P, _DISABLED
    _P = None
    _DISABLED = False
    _draft_cache.clear()
    _warned.clear()


def is_active() -> bool:
    return _P is not None


def _is_draft(prefix: str) -> bool:
    v = _draft_cache.get(prefix)
    if v is None:
        low = prefix.lower()
        v = any(m in low for m in _DRAFT_MARKERS)
        _draft_cache[prefix] = v
    return v


# ------------------------------------------------------------ 1. en el forward


def probe(x: torch.Tensor, layer_idx: int, prefix: str = "",
          hc_count: int = 1) -> None:
    """x (N, hc_count*H): escribe cos(sum_streams(x), r_hat) en la columna de la
    capa. No devuelve nada y NO modifica x: el forward sigue bit-exacto.

    Por que el tronco multi-stream y no `block_input`: block_input sale de una
    RMSNorm con ganancia POR DIMENSION, que deforma la geometria respecto a r_hat
    (hallada en el espacio del residual, donde escriben los writers). La suma de
    los streams es el residual que el modelo lee al final (hc_head suma streams).
    """
    p = _P if _P is not None else _try_init()
    if p is None:
        return
    col = p.col_of.get(int(layer_idx))
    if col is None or _is_draft(prefix):
        return
    k = int(hc_count)
    if x.dim() != 2 or k < 1 or x.shape[-1] != p.hidden * k:
        disable(f"{prefix}: forma {tuple(x.shape)}, se esperaba (N, {k}x{p.hidden})")
        return
    n = x.shape[0]
    if n > p.buf.shape[0]:
        _warn_once("rows", f"lote de {n} tokens > buffer {p.buf.shape[0]}; paso sin sonda")
        return
    xf = x.float()
    if k > 1:
        xf = xf.reshape(n, k, p.hidden).sum(dim=1)
    cos = (xf @ p.dir) / xf.norm(dim=-1).clamp_min(1e-6)
    p.buf[:n, col].copy_(cos)


# ------------------------------------------------------- 2. en el model runner


def step_stats(num_tokens: int, num_reqs: int, query_start_loc: torch.Tensor,
               is_prefilling_np) -> Optional[tuple]:
    """Resume el buffer del paso por peticion: (max, suma, n, ultimo token x K).

    Todo en GPU y sin sincronizar: la unica salida es una copia no bloqueante que
    se completa antes del copy_event del AsyncOutput (ver docstring del modulo).
    """
    p = _P
    if p is None:
        return None
    n = int(num_tokens)
    r = int(num_reqs)
    if n <= 0 or r <= 0 or n > p.buf.shape[0]:
        return None
    try:
        P = p.buf[:n]
        s = P.mean(dim=1)
        qsl = query_start_loc[: r + 1].long()
        ends = qsl[1:]
        tok = torch.arange(n, device=P.device, dtype=torch.long)
        seg = torch.searchsorted(ends, tok, right=True).clamp_(max=r - 1)
        mx = torch.full((r,), -math.inf, device=P.device, dtype=torch.float32)
        mx.scatter_reduce_(0, seg, s, reduce="amax", include_self=True)
        sm = torch.zeros(r, device=P.device, dtype=torch.float32).index_add_(0, seg, s)
        cnt = (ends - qsl[:-1]).float()
        last = P[(ends - 1).clamp(min=0, max=n - 1)]
        out = torch.cat([mx[:, None], sm[:, None], cnt[:, None], last], dim=1)
        pre = [bool(v) for v in list(is_prefilling_np)[:r]]
        return (out.to("cpu", non_blocking=True), pre)
    except Exception as e:
        disable(f"step_stats: {e}")
        return None


def materialize(rp) -> Optional[dict]:
    """Tras el copy_event: tensor CPU -> listas planas (viaja por IPC al scheduler)."""
    if rp is None or isinstance(rp, dict):
        return rp
    try:
        t, pre = rp
        return {"rows": t.tolist(), "prefill": list(pre)}
    except Exception as e:
        _warn_once("materialize", f"materialize: {e}")
        return None


# ----------------------------------------------------------- 3. en el scheduler


def _new_acc(k: int) -> dict:
    return {"p_max": -math.inf, "p_sum": 0.0, "p_n": 0.0, "p_last": None,
            "g_max": -math.inf, "g_sum": 0.0, "g_n": 0.0, "k": k}


def accumulate(request: Any, rp: Optional[dict], idx: int) -> None:
    if not rp:
        return
    try:
        rows = rp["rows"]
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]
        mx, sm, cnt = float(row[0]), float(row[1]), float(row[2])
        last = [float(v) for v in row[3:]]
        pre = bool(rp["prefill"][idx]) if idx < len(rp["prefill"]) else False
        acc = getattr(request, "_refusal_probe_acc", None)
        if acc is None:
            acc = _new_acc(len(last))
            request._refusal_probe_acc = acc
        if pre:
            acc["p_max"] = max(acc["p_max"], mx)
            acc["p_sum"] += sm
            acc["p_n"] += cnt
            # El ultimo token del ULTIMO tramo de prefill es el ultimo del prompt
            # (vLLM siempre recalcula al menos ese aunque haya cache de prefijo).
            acc["p_last"] = last
        elif acc["g_n"] < GEN_WINDOW:
            acc["g_max"] = max(acc["g_max"], mx)
            acc["g_sum"] += sm
            acc["g_n"] += cnt
    except Exception as e:
        _warn_once("accumulate", f"accumulate: {e}")


# Calibracion: JSON {"id", "feature", "threshold", "op": ">=" | "<="}. Se relee
# por mtime (como mucho cada 5 s): el ConfigMap montado se actualiza sin reiniciar
# el residente, que es lo que hace barato recalibrar.
_cal_cache: dict = {"path": None, "mtime": None, "checked": 0.0, "cal": None}


def _load_cal() -> Optional[dict]:
    path = os.environ.get(_ENV_CAL)
    if not path:
        return None
    now = time.monotonic()
    c = _cal_cache
    if c["path"] == path and now - c["checked"] < 5.0:
        return c["cal"]
    c["checked"] = now
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        c.update(path=path, mtime=None, cal=None)
        return None
    if c["path"] == path and c["mtime"] == mtime:
        return c["cal"]
    cal = None
    try:
        raw = json.loads(open(path, encoding="utf-8").read())
        feature = raw.get("feature")
        thr = float(raw["threshold"])
        op = raw.get("op", ">=")
        if feature in FEATURES and op in (">=", "<=") and math.isfinite(thr) and raw.get("id"):
            cal = {"id": str(raw["id"]), "feature": feature, "threshold": thr, "op": op}
        else:
            _warn_once(f"cal:{mtime}", f"calibracion {path} invalida: {raw!r}")
    except Exception as e:
        _warn_once(f"cal:{mtime}", f"calibracion {path} ilegible: {e}")
    c.update(path=path, mtime=mtime, cal=cal)
    return cal


def _r(v: float) -> Optional[float]:
    return round(v, 5) if v is not None and math.isfinite(v) else None


def finalize(request: Any) -> Optional[dict]:
    """Veredicto de la peticion. None si la sonda no vio ningun token suyo."""
    acc = getattr(request, "_refusal_probe_acc", None)
    if acc is None:
        return None
    try:
        p_last = acc["p_last"]
        feats = {
            "prompt_last_mean": (sum(p_last) / len(p_last)) if p_last else None,
            "prompt_max": acc["p_max"] if acc["p_n"] else None,
            "prompt_mean": (acc["p_sum"] / acc["p_n"]) if acc["p_n"] else None,
            "gen_max": acc["g_max"] if acc["g_n"] else None,
            "gen_mean": (acc["g_sum"] / acc["g_n"]) if acc["g_n"] else None,
        }
        cal = _load_cal()
        feature = cal["feature"] if cal else os.environ.get(_ENV_FEATURE, DEFAULT_FEATURE)
        if feature not in FEATURES:
            feature = DEFAULT_FEATURE
        score = feats.get(feature)
        flag = None
        if cal is not None and score is not None:
            flag = score >= cal["threshold"] if cal["op"] == ">=" else score <= cal["threshold"]
        p = _P
        return {
            "v": 1,
            "score": _r(score),
            "feature": feature,
            "flag": flag,
            "cal": cal["id"] if cal else None,
            "layers": list(p.layers) if p is not None else list(parse_layers(os.environ.get(_ENV_LAYERS))),
            "prompt_last": [_r(v) for v in p_last] if p_last else None,
            **{k: _r(v) for k, v in feats.items() if k != "prompt_last_mean"},
            "prompt_last_mean": _r(feats["prompt_last_mean"]),
            "prompt_n": int(acc["p_n"]),
            "gen_n": int(acc["g_n"]),
        }
    except Exception as e:
        _warn_once("finalize", f"finalize: {e}")
        return None
    finally:
        try:
            del request._refusal_probe_acc
        except Exception:
            pass


__all__ = [
    "accumulate", "disable", "enabled_by_env", "finalize", "init_eager", "init_with_direction",
    "is_active", "materialize", "parse_layers", "probe", "reset", "step_stats",
]
