#!/usr/bin/env python3
"""DGX-847 · vista TP=3 del checkpoint NVFP4 de ursuciprian.

Fuente (descargada en el x86, sin tocar los Sparks):
  ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE rev 16c9bd54788d12390838a65ce4a4ecda97fa5f1d
  local-inference-lab/Qwen3.8-Flash-Next-NVFP4 rev 7c4f1bc1 -> solo model-00035-of-00036.safetensors + index
     (es la copia MXFP8 de las capas W4A16; `lm_head.weight` cae en el shard 00034 y no tiene
      `weight_scale`, asi que la copia MXFP8 del lm_head NO sale de aqui: es cuantizacion online del
      lm_head bf16 del checkpoint principal, vocab_parallel_embedding.py:365-395.)

Herencia: convert-tp3-base.py (nuestro convert.py de DGX-780, para el checkpoint FP8 hibrido). Copiada la
maquinaria genica (cabeceras safetensors en crudo, `gather` por unidades, `write_file` con tmp+rename y
sha256) y cambiadas las reglas, que son lo que aqui difiere:

  * formatos de escala: NVFP4 (grupo 16, peso empaquetado 2 por byte, escala F8_E4M3 una cada 16, mas un
    `weight_scale_2` F32 scalar que no se toca) y MXFP8 (bloque [1,32]: peso F8_E4M3 byte por elemento y
    escala U8 E8M0 una cada 32 columnas, filas 1:1). El relleno: peso 0 y escala 1 (0x38 en E4M3, 0x7f en
    E8M0, que es 2^(127-127)=1 y no el 0xff reservado).
  * intermedio de los expertos dirigidos 640 -> 768 (por rango 256, multiplo de 128: es la via mejor
    soportada por los kernels b12x; con 672 por rango 224 solo se cumple %32). Hace falta porque su
    `--moe-backend b12x` rechaza expert-parallel (fused_moe/b12x.py:568-572) y sin EP el intermedio se
    parte por TP (fused_moe/config.py:1349).
  * `vocab_size` se queda 248320 en el config de la vista: su fork padrea a math.lcm(64, tp)=192
    (vocab_parallel_embedding.py:336) y enmascara el relleno con -inf (logits_processor.py:361/389/441).
    Los tensores si se padrean a 248448 filas, porque la copia MXFP8 corta por rangos contra
    `output_size_per_partition` (kernels/linear/nvfp4/b12x.py:95-155).
  * PLE: `padded_vocab_size` 320.001.536 no divide por 3, asi que el alineamiento sube a 384 -> 320.001.792
    y cada una de las 128 partes gana 2 filas (2.500.012 -> 2.500.014). AQUI la tabla si se reescribe: el
    loader de su fork exige la forma exacta por parte (ple_layer.py:845-849) y cobertura completa del rango
    (:624-635), asi que el QWEN_TP3_PLE_SHARD_SKIP de nuestro launch.sh no vale.

Todo es copia de bytes: no se recuantiza nada y las filas/columnas viajan enteras con sus escalas.

Uso (stdlib puro; ejecutar desde un cwd neutro con `python3 -I`):
  convert-ursu-tp3.py plan     --src DIR [--src2 DIR2]              # que se toca y como, sin escribir
  convert-ursu-tp3.py convert  --src DIR --out DIR [--src2 D --out2 D]
  convert-ursu-tp3.py shapes   --view DIR [--view2 DIR2]            # la comprobacion TP=3 de las formas
  convert-ursu-tp3.py verify   --src DIR --out DIR [--src2 D --out2 D]  # equivalencia por bytes (muestreo)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
from pathlib import Path

REVISION = "16c9bd54788d12390838a65ce4a4ecda97fa5f1d"
REVISION2 = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
TP = 3
INDEX = "model.safetensors.index.json"
VOCAB_PADDING = 64  # DEFAULT_VOCAB_PADDING_SIZE de VocabParallelEmbedding
MOE_ALIGN = 128  # pad del intermedio a multiplo de MOE_ALIGN*TP -> por rango multiplo de 128

ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
            "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F4": 1, "F8_E8M0": 1}
# Relleno de una escala que vale 1,0 en su propio formato.
SCALE_ONE = {"F32": struct.pack("<f", 1.0), "F8_E4M3": b"\x38", "U8": b"\x7f", "BF16": b"\x00\x3c",
             "F16": b"\x00\x3c", "I64": b"\x00\x00\x00\x00\x00\x00\x00\x00"}
# Divisor de cada familia a lo largo del eje 0 (filas) y del eje ultimo (columnas logicas).
FAMS = {
    "nvfp4": {"weight": (1, 2), "scale": (1, 16), "scale_dtype": "F8_E4M3"},
    "mxfp8": {"weight": (1, 1), "scale": (1, 32), "scale_dtype": "U8"},
    "none": {"weight": (1, 1), "scale": (1, 1), "scale_dtype": None},
}

# Cifras del config.json de origen (text_config). Si una no casa, se aborta antes de escribir.
EXPECTED = {
    "hidden_size": 2560, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "shared_expert_intermediate_size": 640,
    "moe_intermediate_size": 640, "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
    "vocab_size": 248320, "mtp_num_hidden_layers": 1, "indexer_n_heads": 4, "indexer_kv_heads": 1,
    "indexer_head_dim": 128, "ple_embed_dim": 2560, "ngram_size": 3, "heads_per_ngram": 8,
    "split_ngram_parts": 128, "ngram_vocab_size_base": 20_000_000,
    "make_ngram_vocab_size_divisible_by": 128, "ple_embedding_dtype": "nvfp4",
}


class ConversionError(Exception):
    pass


def text_config(cfg: dict) -> dict:
    return cfg["text_config"] if "text_config" in cfg else cfg


def check_source_config(tc: dict) -> list[str]:
    return [f"{k}: esperado {v!r}, visto {tc.get(k)!r}" for k, v in EXPECTED.items() if tc.get(k) != v]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _lcm(a: int, b: int) -> int:
    return a * b // _gcd(a, b)


# --- PLE ---------------------------------------------------------------------------------------


def ple_view(tc: dict, rows: int, tp: int = TP) -> dict:
    """Filas de la tabla PLE en la vista. `rows` = filas totales del checkpoint (partes ya alineadas al
    divisor de origen). Como el divisor nuevo es multiplo del viejo, ceil(new/128) es lo que pide la clase
    de la imagen sin rehacer sus primos."""
    div, parts = int(tc["make_ngram_vocab_size_divisible_by"]), int(tc["split_ngram_parts"])
    padded = _round_up(rows, div)
    new_div = div if padded % tp == 0 else div * tp // _gcd(div, tp)
    new = _round_up(rows, new_div)
    return {"checkpoint_rows": rows, "divisor": div, "padded_vocab_size": padded, "new_divisor": new_div,
            "new_padded_vocab_size": new, "pad_rows": new - rows, "parts": parts,
            "rows_per_rank": new // tp, "checkpoint_part_rows": -(-rows // parts),
            "image_part_rows": -(-new // parts), "rows_per_part_added": -(-new // parts) - -(-rows // parts)}


# --- Mapa de cabezas --------------------------------------------------------------------------


class Layout:
    """Dimensiones del origen y de la vista TP=3, sacadas del text_config."""

    def __init__(self, tc: dict, tp: int = TP):
        self.tp = tp
        self.head_dim = int(tc["head_dim"])
        self.kv = int(tc["num_key_value_heads"])
        self.kv_new = self.kv * tp // _gcd(self.kv, tp)  # 2 -> 6
        self.gk = int(tc["linear_num_key_heads"])
        self.gv = int(tc["linear_num_value_heads"])
        self.ratio = self.gv // self.gk  # 3
        # La vista TP=3 es la forma por rango del TP=2 de la receta: 8 k y 24 v por rango (24:72 global).
        self.gk_new = self.gk // 2 * tp
        self.gv_new = self.gk_new * self.ratio
        self.kdim = int(tc["linear_key_head_dim"])
        self.vdim = int(tc["linear_value_head_dim"])
        self.shared = int(tc["shared_expert_intermediate_size"])
        self.shared_new = _round_up(self.shared, MOE_ALIGN * tp)
        self.moe = int(tc["moe_intermediate_size"])
        self.moe_new = _round_up(self.moe, MOE_ALIGN * tp)
        self.vocab = int(tc["vocab_size"])
        self.q = int(tc["num_attention_heads"])
        self.vocab_new = _round_up(self.vocab, _lcm(VOCAB_PADDING, tp))  # 248320 -> 248448 (lcm 192)
        self.groups = gdn_group_slots(self.gk, self.gk_new, tp)

    def target_config(self) -> dict:
        """Lo que cambia en el config de la vista. `vocab_size` NO esta: lo padrea el fork."""
        return {
            "num_key_value_heads": self.kv_new,
            "linear_num_key_heads": self.gk_new,
            "linear_num_value_heads": self.gv_new,
            "shared_expert_intermediate_size": self.shared_new,
            "moe_intermediate_size": self.moe_new,
        }

    def per_rank(self) -> dict:
        return {"kv_heads": self.kv_new // self.tp, "gdn_key": self.gk_new // self.tp,
                "gdn_value": self.gv_new // self.tp, "shared": self.shared_new // self.tp,
                "moe": self.moe_new // self.tp, "vocab": self.vocab_new // self.tp,
                "q_heads": self.q // self.tp}


def gdn_group_slots(gk: int, gk_new: int, tp: int) -> list[int | None]:
    """Hueco -> grupo GDN original (None = cabeza a cero). Rango r: sus grupos reales primero."""
    per_rank = gk_new // tp
    base, rest = divmod(gk, tp)
    slots: list[int | None] = []
    g = 0
    for r in range(tp):
        count = base + (r < rest)
        if count > per_rank:
            raise ConversionError(f"GDN: {count} grupos no caben en {per_rank} huecos del rango {r}")
        slots += [g + i if i < count else None for i in range(per_rank)]
        g += count
    return slots


# Unidad = (inicio en el origen o None para relleno, numero de elementos LOGICOS del eje).
def kv_units(L: Layout) -> list:
    rep = L.kv_new // L.kv
    return [((j // rep) * L.head_dim, L.head_dim) for j in range(L.kv_new)]


def qkv_units(L: Layout) -> list:
    k0, v0 = L.gk * L.kdim, 2 * L.gk * L.kdim
    q = [(None if g is None else g * L.kdim, L.kdim) for g in L.groups]
    k = [(None if g is None else k0 + g * L.kdim, L.kdim) for g in L.groups]
    return q + k + vhead_units(L, L.vdim, v0)


def vhead_units(L: Layout, size: int, offset: int = 0) -> list:
    return [
        (None if g is None else offset + (g * L.ratio + j) * size, size)
        for g in L.groups
        for j in range(L.ratio)
    ]


def pad_units(n: int, n_new: int) -> list:
    return [(0, n), (None, n_new - n)]


def _pattern(name: str) -> str:
    """Normaliza numeros de capa/experto/parte: dos tensores del mismo patron son la misma regla."""
    return re.sub(r"\.\d+\.", ".N.", name)


def coalesce(units: list) -> list:
    out: list = []
    for start, n in units:
        if n == 0:
            continue
        if out:
            ps, pn = out[-1]
            if (ps is None and start is None) or (
                ps is not None and start is not None and ps + pn == start
            ):
                out[-1] = (ps, pn + n)
                continue
        out.append((start, n))
    return out


# --- Reglas: que tensor cambia y por donde ------------------------------------------------------
# (regex, kind, eje, es_escala). La familia NVFP4/MXFP8/ninguna NO esta aqui: sale del dtype, porque el
# mismo tensor es NVFP4 en el checkpoint principal y MXFP8 en la copia de la capa grande (y al reves:
# self_attn sale MXFP8 del principal y linear_attn.in_proj_qkv NVFP4).
LIN = r"(?:model\.language_model|mtp)\.layers\.\d+\.linear_attn\."
ATT = r"(?:model\.language_model|mtp)\.layers\.\d+\.self_attn\."
EXPERT = r"(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.\d+\."
SHARED = r"(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.shared_expert\."
RULES = (
    (rf"^{ATT}[kv]_proj\.weight$", "kv", 0, False),
    (rf"^{ATT}[kv]_proj\.weight_scale$", "kv", 0, True),
    (rf"^{LIN}in_proj_qkv\.weight$", "qkv", 0, False),
    (rf"^{LIN}in_proj_qkv\.weight_scale$", "qkv", 0, True),
    (rf"^{LIN}in_proj_z\.weight$", "z", 0, False),
    (rf"^{LIN}in_proj_z\.weight_scale$", "z", 0, True),
    (rf"^{LIN}in_proj_[ab]\.weight$", "ab", 0, False),
    (rf"^{LIN}in_proj_[ab]\.weight_scale$", "ab", 0, True),
    (rf"^{LIN}conv1d\.weight$", "conv", 0, False),
    (rf"^{LIN}(?:A_log|dt_bias)$", "vhead", 0, False),
    (rf"^{LIN}out_proj\.weight$", "out", 1, False),
    (rf"^{LIN}out_proj\.weight_scale$", "out", 1, True),
    (rf"^{SHARED}(?:gate|up)_proj\.weight$", "shared_in", 0, False),
    (rf"^{SHARED}(?:gate|up)_proj\.weight_scale$", "shared_in", 0, True),
    (rf"^{SHARED}down_proj\.weight$", "shared_down", 1, False),
    (rf"^{SHARED}down_proj\.weight_scale$", "shared_down", 1, True),
    (rf"^{EXPERT}(?:gate|up)_proj\.weight$", "moe_in", 0, False),
    (rf"^{EXPERT}(?:gate|up)_proj\.weight_scale$", "moe_in", 0, True),
    (rf"^{EXPERT}down_proj\.weight$", "moe_down", 1, False),
    (rf"^{EXPERT}down_proj\.weight_scale$", "moe_down", 1, True),
    (r"^(?:model\.language_model\.embed_tokens|lm_head)\.weight$", "vocab", 0, False),
    (r"^(?:model\.language_model\.embed_tokens|lm_head)\.weight_scale$", "vocab", 0, True),
    (r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$", "ple", 0, False),
    (r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight_scale$", "ple", 0, True),
)
# Lo que NO debe cambiar ni un byte, pero lo enumeramos para que un nombre nuevo salte en el plan.
UNTOUCHED = (
    rf"^{ATT}(?:q|o)_proj\.weight(?:_scale|_scale_2)?$",
    rf"^{ATT}indexer\.(?:index_qk_proj\.weight(?:_scale)?|q_layernorm\.weight|k_layernorm\.weight)$",
    r"(?:_norm|norm|layernorm)\.weight$",
    r"\.mlp\.(?:gate|shared_expert_gate)\.weight$",
    r"\.input_scale$",
    r"\.weight_scale_2$",
    r"hyper_connection",
    r"\.ple\.(?:conv1d|key_proj|value_proj|norm_\w+)\.",  # rama PLE replicada (ple_layer.py:971-992)
    r"\.ple\.ple_embedding\.(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes|weight_scale_2)$",
    r"^mtp\.(?:fc|enorm|hnorm|pre_fc_norm|norm)",
    r"\.visual\.",
)


def family_of(dtype: str, is_scale: bool) -> str:
    """NVFP4 = peso U8 empaquetado 2/byte con escala F8_E4M3 cada 16; MXFP8 = peso F8_E4M3 con escala U8
    cada 32; `none` = sin cuantizar (bf16). Los dos checkpoints mezclan familias capa a capa."""
    if is_scale:
        fam = {"F8_E4M3": "nvfp4", "U8": "mxfp8"}.get(dtype)
    else:
        fam = {"U8": "nvfp4", "F8_E4M3": "mxfp8", "BF16": "none", "F16": "none"}.get(dtype)
    if fam is None:
        raise ConversionError(f"dtype {dtype} ({'escala' if is_scale else 'peso'}) sin familia")
    return fam


def units_for(kind: str, L: Layout) -> list:
    if kind == "kv":
        return kv_units(L)
    if kind in ("qkv", "conv"):
        return qkv_units(L)
    if kind in ("z", "out"):
        return vhead_units(L, L.vdim)
    if kind in ("ab", "vhead"):
        return vhead_units(L, 1)
    if kind == "shared_in" or kind == "shared_down":
        return pad_units(L.shared, L.shared_new)
    if kind in ("moe_in", "moe_down"):
        return pad_units(L.moe, L.moe_new)
    if kind == "vocab":
        return pad_units(L.vocab, L.vocab_new)
    raise ConversionError(kind)


def to_tensor_units(units: list, fam: str, is_scale: bool, axis: int) -> list:
    """Unidades logicas -> unidades en el espacio del propio tensor (packed y granularidad de escala)."""
    rows_div, cols_div = FAMS[fam]["weight" if not is_scale else "scale"]
    div = rows_div if axis == 0 else cols_div
    if div == 1:
        return units
    out = []
    for start, n in units:
        if n % div:
            raise ConversionError(f"unidad ({start}, {n}) no es multiplo de {div} en {fam}")
        if start is not None and start % div:
            raise ConversionError(f"unidad ({start}, {n}) empieza desfasada por {div} en {fam}")
        out.append((None if start is None else start // div, n // div))
    return out


# --- safetensors en crudo (cabecera + bytes) ---------------------------------------------------


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def gather(base: int, shape: list, dtype: str, axis: int, units: list, fill: bytes) -> tuple[list, list]:
    """Reordena por bloques a lo largo de `axis` el tensor que empieza en el byte `base` del fichero."""
    isz = ITEMSIZE[dtype]
    outer = 1
    for d in shape[:axis]:
        outer *= d
    inner = 1
    for d in shape[axis + 1:]:
        inner *= d
    cols = shape[axis]
    units = coalesce(units)
    for start, n in units:
        if start is not None and (start < 0 or start + n > cols):
            raise ConversionError(f"unidad ({start}, {n}) fuera de {cols} en el eje {axis}")
    if len(fill) != isz:
        raise ConversionError(f"relleno de {len(fill)} bytes para {dtype} ({isz})")
    unit = inner * isz
    chunks: list = []
    for o in range(outer):
        row = base + o * cols * unit
        for start, n in units:
            if start is None:
                chunks.append(fill * (n * inner))
            elif chunks and isinstance(chunks[-1], tuple) and chunks[-1][1] == row + start * unit:
                chunks[-1] = (chunks[-1][0], row + (start + n) * unit)
            else:
                chunks.append((row + start * unit, row + (start + n) * unit))
    return shape[:axis] + [sum(n for _, n in units)] + shape[axis + 1:], chunks


def chunk_len(c) -> int:
    return len(c) if isinstance(c, bytes) else c[-1] - c[-2]


class Plan:
    """Que ficheros se reescriben, que tensores cambian y como (de un checkpoint)."""

    def __init__(self, src: Path, cfg: dict, tp: int = TP, label: str = "principal"):
        self.src = src
        self.label = label
        self.cfg = cfg
        self.tc = text_config(cfg)
        self.L = Layout(self.tc, tp)
        wm = json.loads((src / INDEX).read_text())["weight_map"]
        # Del segundo checkpoint solo se baja el shard de las copias MXFP8: el index declara 36 ficheros y
        # aqui solo hay uno. Fuera de esa etiqueta, que falte un shard es un error.
        missing = {f for f in set(wm.values()) if not (src / f).exists()}
        if missing and label != "mxfp8":
            raise ConversionError(f"{label}: faltan {len(missing)} shards, p.e. {sorted(missing)[:3]}")
        self.weight_map = {k: v for k, v in wm.items() if v not in missing}
        self.headers: dict = {}
        for name in sorted(set(self.weight_map.values())):
            self.headers[name] = read_header(src / name)
        self.changes: dict = {}  # tensor -> (kind, eje, unidades-en-espacio-del-tensor, relleno)
        self.kinds: dict = {}  # tensor -> (kind, fam, is_scale) para el informe
        self.ple_shapes: dict = {}
        self.ple: dict = {}
        self._classify()
        self.rewrite = sorted({self.weight_map[t] for t in self.changes})

    def _classify(self) -> None:
        """Regla por tensor. Un nombre que no casa ni con una regla ni con la lista de intocables aborta:
        es la puerta de completitud (lo que no se clasifica no se convierte, y eso en TP=3 es un hang)."""
        seen: dict = {}
        unknown: dict = {}
        for tensor, fname in sorted(self.weight_map.items()):
            info = self.headers[fname][0].get(tensor)
            if info is None:
                raise ConversionError(f"{tensor} declarado en el index pero no en {fname}")
            hit = next((r for r in RULES if re.search(r[0], tensor)), None)
            if hit and hit[1] == "ple":  # la tabla se planifica despues de conocer ple_view
                self.ple_shapes[tensor] = list(info["shape"])
                continue
            if hit:
                _rx, kind, axis, is_scale = hit
                fam = family_of(info["dtype"], is_scale)
                t_units = to_tensor_units(units_for(kind, self.L), fam, is_scale, axis)
                fill = self._fill(info["dtype"], is_scale, fam, tensor)
                self._add(tensor, info, kind, axis, t_units, fill)
                self.kinds[tensor] = (kind, fam, is_scale)
                if not is_scale:
                    seen[kind] = seen.get(kind, 0) + 1
            elif any(re.search(rx, tensor) for rx in UNTOUCHED):
                continue
            else:
                unknown[_pattern(tensor)] = tensor
        if unknown:
            raise ConversionError("nombres sin clasificar: "
                                  + ", ".join(sorted(unknown)[:8]) + f" ({len(unknown)} en total)")
        self._ple_pass(seen)
        self._check_counts(seen)

    def _ple_pass(self, seen: dict) -> None:
        if not self.ple_shapes:
            return
        parts = int(self.tc["split_ngram_parts"])
        weights = [t for t in self.ple_shapes if t.endswith(".weight")]
        if len(weights) != parts:
            raise ConversionError(f"PLE: {len(weights)} partes y split_ngram_parts {parts}")
        rows = {tuple(self.ple_shapes[t]) for t in weights}
        if len(rows) != 1:
            raise ConversionError(f"PLE: partes con formas distintas {sorted(rows)}")
        total = sum(self.ple_shapes[t][0] for t in weights)
        self.ple = ple_view(self.tc, total, self.L.tp)
        if not self.ple["rows_per_part_added"]:
            return  # el padded ya divida: ni se tocan las partes ni se sube el divisor
        for tensor in sorted(self.ple_shapes):
            info = self.headers[self.weight_map[tensor]][0][tensor]
            is_scale = tensor.endswith("_scale")
            fam = family_of(info["dtype"], is_scale)
            self.changes[tensor] = ("ple", 0, int(tensor.rsplit(".weight")[0].rsplit("shard_", 1)[1]),
                                    self._fill(info["dtype"], is_scale, fam, tensor))
            self.kinds[tensor] = ("ple", fam, is_scale)
            if not is_scale:
                seen["ple"] = seen.get("ple", 0) + 1

    def _fill(self, dtype: str, is_scale: bool, fam: str, tensor: str) -> bytes:
        if dtype not in ITEMSIZE:
            raise ConversionError(f"{tensor}: dtype {dtype} sin tamano")
        if not is_scale:
            return bytes(ITEMSIZE[dtype])
        if dtype not in SCALE_ONE:
            raise ConversionError(f"{tensor}: escala {dtype} sin relleno conocido")
        want = FAMS[fam]["scale_dtype"]
        if want and dtype != want:
            raise ConversionError(f"{tensor}: escala {dtype}, la familia {fam} espera {want}")
        return SCALE_ONE[dtype]

    def _add(self, tensor, info, kind, axis, units, fill) -> None:
        shape = list(info["shape"])
        size = shape[axis]
        real = sorted((s, n) for s, n in units if s is not None)
        if kind == "kv":  # duplicado: cada cabeza original sale `rep` veces
            step = real[0][1]
            ok = set(real) == {(i * step, step) for i in range(size // step)} and size % step == 0
        else:  # permutacion o pad: las unidades reales cubren el eje una sola vez
            ok = ([s for s, _ in real] == [sum(n for _, n in real[:i]) for i in range(len(real))]
                  and sum(n for _, n in real) == size)
        if not ok:
            raise ConversionError(f"{tensor}: forma {shape} no casa con {kind} (eje {axis}, unidades {units[:3]}...)")
        self.changes[tensor] = (kind, axis, units, fill)

    def ple_emit(self, tensor: str, info: dict, part: int, fill: bytes) -> tuple[list, list]:
        """Parte `part` de la vista = filas globales [part*filas_vista, +filas_vista) de la tabla padreada,
        leidas de las partes del checkpoint, que tienen OTRO tamano: las fronteras se desplazan 2 filas por
        parte y una parte de la vista puede nacer en dos ficheros del checkpoint (ple_layer.py:826-831 exige
        `shard_size = ceil(padded/split_ngram_parts)` filas exactas por fichero)."""
        p, cpr = self.ple, self.ple["checkpoint_part_rows"]
        first = part * p["image_part_rows"]
        rows = min(p["image_part_rows"], p["new_padded_vocab_size"] - first)
        rb = ITEMSIZE[info["dtype"]]
        for d in info["shape"][1:]:
            rb *= d
        chunks: list = []
        got = 0
        r = first
        while got < rows and r < p["checkpoint_rows"]:
            src_part = r // cpr
            off = r % cpr
            n = min(cpr - off, p["checkpoint_rows"] - r, rows - got)
            sname = tensor.replace(f"shard_{part}.", f"shard_{src_part}.")
            fname = self.weight_map[sname]
            sinfo = self.headers[fname][0][sname]
            if list(sinfo["shape"]) != list(info["shape"]) or sinfo["dtype"] != info["dtype"]:
                raise ConversionError(f"PLE: {sname} no es igual que {tensor}")
            base = self.headers[fname][1] + sinfo["data_offsets"][0]
            chunks.append((fname, base + off * rb, base + (off + n) * rb))
            got += n
            r += n
        if got < rows:
            chunks.append(fill * ((rows - got) // len(fill)))
        return [rows] + list(info["shape"][1:]), chunks

    def emit(self, tensor: str, fname: str, data_start: int) -> tuple[list, list]:
        """(forma, trozos) del tensor en la vista. Un solo camino para escribir y para verificar.
        Trozo = bytes (relleno) | (inicio, fin) (fichero propio) | (fichero, inicio, fin) (cruzado)."""
        info = self.headers[fname][0][tensor]
        if tensor not in self.changes:
            b, e = info["data_offsets"]
            return list(info["shape"]), [(data_start + b, data_start + e)]
        kind, axis, units, fill = self.changes[tensor]
        if kind == "ple":
            return self.ple_emit(tensor, info, units, fill)
        b, e = info["data_offsets"]
        return gather(data_start + b, list(info["shape"]), info["dtype"], axis, units, fill)

    def _check_counts(self, seen: dict) -> None:
        for tensor, (kind, fam, is_scale) in self.kinds.items():
            if is_scale and tensor[: -len("_scale")] not in self.changes:
                raise ConversionError(f"{tensor} sin su peso")
            if (not is_scale and FAMS[fam]["scale_dtype"]
                    and (tensor + "_scale") in self.weight_map
                    and (tensor + "_scale") not in self.changes):
                raise ConversionError(f"{tensor} con escala sin regla de escala")
        if self.label != "principal":
            return  # del segundo checkpoint solo hay un shard: la puerta es mxfp8_copy_check
        want = expected_counts(self.tc, bool(self.ple) and self.ple["rows_per_part_added"] > 0)
        if seen != want:
            raise ConversionError(f"tensores tocados {dict(sorted(seen.items()))} != esperados "
                                  f"{dict(sorted(want.items()))}")

    def config(self) -> dict:
        cfg = json.loads(json.dumps(self.cfg))
        tc = text_config(cfg)
        tc.update(self.L.target_config())
        if self.ple:
            tc["make_ngram_vocab_size_divisible_by"] = self.ple["new_divisor"]
        return cfg

    def output_bytes(self, fname: str) -> int:
        """Bytes de datos (sin cabecera) que tendra `fname` en la vista."""
        header, data_start = self.headers[fname]
        total = 0
        for tensor in header:
            if tensor == "__metadata__":
                continue
            shape, _chunks = self.emit(tensor, fname, data_start)
            n = ITEMSIZE[header[tensor]["dtype"]]
            for d in shape:
                n *= d
            total += n
        return total

    def index(self) -> dict:
        """El index de la vista: mismos nombres y shards; solo cambia metadata.total_size."""
        total = sum(self.output_bytes(f) for f in sorted(self.headers))
        return {"metadata": {"total_size": total}, "weight_map": self.weight_map}

    def report(self) -> dict:
        by_kind: dict = {}
        for tensor, (kind, fam, is_scale) in self.kinds.items():
            by_kind.setdefault(f"{kind}/{fam}", 0)
            by_kind[f"{kind}/{fam}"] += 1
        return {"checkpoint": self.label, "files_rewritten": len(self.rewrite),
                "tensors_changed": len(self.changes), "by_kind": by_kind,
                "target_config": self.L.target_config(), "per_rank": self.L.per_rank(),
                "gdn_slots": self.L.groups, "ple": self.ple,
                "bytes_out": sum(self.output_bytes(f) for f in self.rewrite),
                "bytes_vista": sum(self.output_bytes(f) for f in self.headers)}


def expected_counts(tc: dict, ple_rewritten: bool) -> dict:
    """Cuantos TENSORES BASE (peso, sin contar su escala) de cada tipo tiene que tocar la conversion,
    derivado del config. Es la puerta numerica: si una capa no tiene shared expert, o el MTP no tiene
    expertos, o una regla casa de mas, salta aqui. Se cuentan bases porque el numero de escalas depende
    de la familia de cada capa, que cambia entre los dos checkpoints."""
    types = tc["layer_types"]
    full = sum(t == "full_attention" for t in types)
    gdn = sum(t == "linear_attention" for t in types)
    mtp = int(tc.get("mtp_num_hidden_layers", 0))
    layers = len(types) + mtp  # el MTP repite capa completa (mtp.layer_types = ['full_attention'])
    experts = int(tc["num_experts"])
    parts = int(tc["split_ngram_parts"])
    return {
        "kv": 2 * (full + mtp),
        "qkv": gdn, "z": gdn, "out": gdn, "ab": 2 * gdn,
        "conv": gdn, "vhead": 2 * gdn,   # conv1d, A_log y dt_bias: sin escala
        "shared_in": 2 * layers, "shared_down": layers,
        "moe_in": 2 * experts * layers, "moe_down": experts * layers,
        "vocab": 2,                      # embed_tokens y lm_head
        "ple": parts if ple_rewritten else 0,
    }


def header_json(header: dict) -> bytes:
    return json.dumps(header, separators=(",", ":")).encode()


def write_file(plan: Plan, fname: str, out: Path, piece: int = 64 << 20) -> tuple[str, int]:
    """Escribe `fname` en `out` (tmp + rename). Devuelve (sha256, bytes). Los trozos pueden venir de otro
    fichero (las partes de la PLE se re-particionan cruzando shards)."""
    header, data_start = plan.headers[fname]
    meta = header.get("__metadata__")
    tensors = sorted((t for t in header if t != "__metadata__"), key=lambda t: header[t]["data_offsets"][0])
    parts = []
    new_header: dict = {} if meta is None else {"__metadata__": meta}
    offset = 0
    for t in tensors:
        shape, chunks = plan.emit(t, fname, data_start)
        size = sum(chunk_len(c) for c in chunks)
        if offset % ITEMSIZE[header[t]["dtype"]]:
            raise ConversionError(f"{fname}:{t} desalineado")
        new_header[t] = {"dtype": header[t]["dtype"], "shape": shape, "data_offsets": [offset, offset + size]}
        parts += chunks
        offset += size
    hj = header_json(new_header)
    hj += b" " * (-len(hj) % 8)
    tmp = out / (fname + ".tmp")
    digest = hashlib.sha256()
    maps: dict = {}
    try:
        with open(tmp, "wb") as fh:
            for data in (struct.pack("<Q", len(hj)), hj):
                fh.write(data)
                digest.update(data)
            for c in parts:
                if isinstance(c, bytes):
                    fh.write(c)
                    digest.update(c)
                    continue
                src_file, a, b = (fname, c[0], c[1]) if len(c) == 2 else c
                if src_file not in maps:
                    fh_ = open(plan.src / src_file, "rb")
                    maps[src_file] = (fh_, mmap.mmap(fh_.fileno(), 0, access=mmap.ACCESS_READ))
                _, mm = maps[src_file]
                for x in range(a, b, piece):
                    data = mm[x:min(b, x + piece)]
                    fh.write(data)
                    digest.update(data)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        for _, mm in maps.values():
            mm.close()
    written = tmp.stat().st_size
    os.replace(tmp, out / fname)
    return digest.hexdigest(), written


def convert_one(src: Path, out: Path, cfg: dict, label: str, copy_rest: bool = True) -> dict:
    plan = Plan(src, cfg, TP, label)
    out.mkdir(parents=True, exist_ok=True)
    to_write = sum(plan.output_bytes(f) for f in plan.rewrite)
    free = shutil.disk_usage(out).free
    if free < 1.2 * to_write:
        raise ConversionError(f"{label}: espacio {free} libres < 1,2 x {to_write}")
    sums: dict = {}
    for fname in plan.rewrite:
        sums[fname], _ = write_file(plan, fname, out)
        print(f"[{label}] reescrito {fname}", flush=True)
    cfg_bytes = (json.dumps(plan.config(), indent=2, ensure_ascii=False) + "\n").encode()
    (out / "config.json").write_bytes(cfg_bytes)
    sums["config.json"] = hashlib.sha256(cfg_bytes).hexdigest()
    if label == "mxfp8":
        # El index del segundo checkpoint se copia tal cual: el loader solo mira el weight_map
        # (kernels/linear/nvfp4/b12x.py:118) y lo que declara son los nombres ya padreados de la vista.
        shutil.copyfile(src / INDEX, out / INDEX)
        sums[INDEX] = _sha256(out / INDEX)
    else:
        idx_bytes = (json.dumps(plan.index(), indent=2) + "\n").encode()
        (out / INDEX).write_bytes(idx_bytes)
        sums[INDEX] = hashlib.sha256(idx_bytes).hexdigest()
    if copy_rest:
        for f in sorted(p for p in src.iterdir() if p.is_file()):
            if f.name in sums or f.name in (INDEX, "SHA256SUMS"):
                continue
            shutil.copyfile(f, out / f.name)
            before, after = _sha256(f), _sha256(out / f.name)
            if before != after:
                raise ConversionError(f"copia corrupta: {f.name}")
            sums[f.name] = after
    (out / "SHA256SUMS").write_text("".join(f"{sums[f]}  {f}\n" for f in sorted(sums)))
    rep = plan.report()
    (out / f"tp3-plan-{label}.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"[{label}] {rep['files_rewritten']} ficheros, {rep['tensors_changed']} tensores, "
          f"{round(rep['bytes_out']/2**30, 2)} GiB de salida", flush=True)
    return rep


def convert(src: Path, out: Path, src2: Path | None, out2: Path | None) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    bad = check_source_config(text_config(cfg))
    if bad:
        raise ConversionError("config.json de origen no casa:\n  " + "\n  ".join(bad))
    reps = {"principal": convert_one(src, out, cfg, "principal")}
    if src2 and out2:
        # El segundo checkpoint no tiene config.json: se le pasa el del principal.
        reps["mxfp8"] = convert_one(src2, out2, cfg, "mxfp8", copy_rest=False)
    (out / "tp3-vista.json").write_text(json.dumps({
        "revision": REVISION, "revision_mxfp8": REVISION2, "tp": TP, "informes": reps}, indent=2) + "\n")
    return reps


# --- shapes: la comprobacion TP=3 de la vista, en puro python ----------------------------------


def divide(a: int, b: int, que: str) -> int:
    if a % b:
        raise ConversionError(f"{que}: {a} no divide por {b}")
    return a // b


def expected_axis(tc: dict, kind: str) -> int:
    """Longitud logica del eje que parte la regla, segun el config de la vista."""
    hd, kdim = int(tc["head_dim"]), int(tc["linear_key_head_dim"])
    vdim = int(tc["linear_value_head_dim"])
    gk, gv = int(tc["linear_num_key_heads"]), int(tc["linear_num_value_heads"])
    if kind == "kv":
        return int(tc["num_key_value_heads"]) * hd
    if kind in ("qkv", "conv"):
        return 2 * gk * kdim + gv * vdim
    if kind in ("z", "out"):
        return gv * vdim
    if kind in ("ab", "vhead"):
        return gv
    if kind in ("shared_in", "shared_down"):
        return int(tc["shared_expert_intermediate_size"])
    if kind in ("moe_in", "moe_down"):
        return int(tc["moe_intermediate_size"])
    if kind == "vocab":
        return _round_up(int(tc["vocab_size"]), _lcm(VOCAB_PADDING, TP))
    raise ConversionError(kind)


def ple_geometry(view: Path, tc: dict) -> dict:
    """La geometria que su fork va a calcular: `padded = align_up(sum(prime_sizes), table_alignment)` y
    despues `padded % tp` (ple_hash/geometry.py:95 y _contracts.py:375-380). Leida del I64 del checkpoint."""
    h = _headers(view)
    name = next((t for t in h if t.endswith("ngram_heads_vocab_sizes")), None)
    if name is None:
        raise ConversionError("falta ngram_heads_vocab_sizes en la vista")
    fname = json.loads((view / INDEX).read_text())["weight_map"][name]
    header, data_start = read_header(view / fname)
    b, e = header[name]["data_offsets"]
    raw = _read_at(view / fname, data_start + b, e - b)
    primes = [int(x) for x in struct.unpack(f"<{len(raw) // 8}q", raw)]
    total = sum(primes)
    align = int(tc["make_ngram_vocab_size_divisible_by"])
    parts = int(tc["split_ngram_parts"])
    padded = _round_up(total, align)
    rows = [i["shape"][0] for t, i in h.items() if re.search(r"ngram_embedding\.shard_\d+\.weight$", t)]
    out = {"heads": len(primes), "filas_reales": total, "divisor": align, "padded_vocab_size": padded,
           "padded_entre_tp": padded // TP if padded % TP == 0 else None, "partes": parts,
           "filas_por_parte": -(-padded // parts), "partes_vistas": len(rows),
           "filas_vistas": sorted(set(rows))}
    if padded % TP:
        raise ConversionError(f"PLE: padded_vocab_size {padded} no divide por {TP}")
    if out["filas_por_parte"] != (rows[0] if rows else 0) or len(set(rows)) != 1:
        raise ConversionError(f"PLE: las partes miden {sorted(set(rows))} y la imagen pide "
                              f"{out['filas_por_parte']}")
    if len(rows) != parts:
        raise ConversionError(f"PLE: {len(rows)} partes vistas y split_ngram_parts {parts}")
    if sum(rows) != padded:
        raise ConversionError(f"PLE: las partes suman {sum(rows)} y la tabla es {padded}")
    return out


def shapes(view: Path, view2: Path | None = None) -> dict:
    """Reproduce las reglas de su fork sobre las cabeceras de la vista. Sin GPU y sin torch."""
    tc = text_config(json.loads((view / "config.json").read_text()))
    checks: dict = {}
    # 1) QKVParallelLinear (linear.py:1079-1085) y la guarda de QSA (qsa.py:736-745)
    divide(int(tc["num_attention_heads"]), TP, "cabezas QSA query")
    kv = int(tc["num_key_value_heads"])
    checks["qsa_kv"] = {"kv_por_rango": divide(kv, TP, "cabezas QSA KV"), "replicas": 1}
    # 2) GDN (qwen_gdn_linear_attn.py:753/758 y 871-872)
    checks["gdn"] = {"key_por_rango": divide(int(tc["linear_num_key_heads"]), TP, "GDN key"),
                     "value_por_rango": divide(int(tc["linear_num_value_heads"]), TP, "GDN value")}
    # 3) shared expert (linear.py:738) y MoE (fused_moe/config.py:1349, b12x.py:500-520)
    sh = divide(int(tc["shared_expert_intermediate_size"]), TP, "shared expert")
    m_per = divide(int(tc["moe_intermediate_size"]), TP, "moe intermediate")
    checks["shared"] = {"por_rango": sh, "alineado_32": sh % 32 == 0}
    checks["moe"] = {"por_rango": m_per, "alineado_32": m_per % 32 == 0, "alineado_128": m_per % 128 == 0,
                     "hidden_alineado_256": int(tc["hidden_size"]) % 256 == 0}
    # 4) vocab: VocabParallelEmbedding padrea a math.lcm(64, tp) y el config conserva el valor real
    pad = _lcm(VOCAB_PADDING, TP)
    vocab_pad = _round_up(int(tc["vocab_size"]), pad)
    checks["vocab"] = {"config_vocab_size": int(tc["vocab_size"]), "lcm_padding": pad,
                       "num_embeddings_padded": vocab_pad, "por_rango": divide(vocab_pad, TP, "vocab"),
                       "por_rango_alineado_32": vocab_pad // TP % 32 == 0}
    # 5) PLE: la geometria que calcula el fork, contra las partes de la vista
    checks["ple"] = ple_geometry(view, tc)
    # 6) draft vocab del MTP (mtp.py:684): padrea solo
    checks["draft_vocab"] = {"K": 131072, "padded": _round_up(131072, pad),
                             "por_rango": _round_up(131072, pad) // TP}
    # 7) puerta de formas: todo tensor que toca una regla mide en la vista lo que anuncia el config
    h = _headers(view)
    tocados: dict = {}
    bad = []
    for t, info in h.items():
        if info["dtype"] not in ITEMSIZE:
            bad.append(f"{t}: dtype {info['dtype']}")
        hit = next((r for r in RULES if re.search(r[0], t)), None)
        if not hit or hit[1] == "ple":
            continue
        _rx, kind, axis, is_scale = hit
        fam = family_of(info["dtype"], is_scale)
        div = FAMS[fam]["weight" if not is_scale else "scale"][axis]
        want = expected_axis(tc, kind) // div
        tocados[kind] = tocados.get(kind, 0) + 1
        if list(info["shape"])[axis] != want:
            bad.append(f"{t}: eje {axis} mide {info['shape'][axis]} y la vista pide {want}")
    checks["formas"] = {"tensores": len(h), "comprobados": tocados, "problemas": bad[:10]}
    if bad:
        raise ConversionError("formas: " + "; ".join(bad[:5]))
    # 8) la copia MXFP8 del segundo checkpoint: mismo corte que _tp_shard
    if view2:
        checks["mxfp8_copy"] = mxfp8_copy_check(view, view2)
    checks["pasa"] = True
    return checks


def _headers(view: Path, present_only: bool = False) -> dict:
    """Todas las cabeceras de un directorio de vista, por nombre de tensor."""
    out: dict = {}
    wm = json.loads((view / INDEX).read_text())["weight_map"]
    for fname in sorted(set(wm.values())):
        if present_only and not (view / fname).exists():
            continue
        header, _ = read_header(view / fname)
        for t, info in header.items():
            if t != "__metadata__":
                out[t] = info
    return out


def logical_shape(info: dict) -> tuple[int, int]:
    """(filas, columnas logicas) de un peso: un NVFP4 empaquetado guarda 2 valores por byte."""
    shape = list(info["shape"])
    cols = shape[-1]
    if family_of(info["dtype"], False) == "nvfp4":
        cols *= 2
    return shape[0], cols


def scale_ok(info: dict, weight: dict) -> bool:
    """La escala tiene que tener 1 fila por fila logica y una columna por bloque (16 NVFP4, 32 MXFP8)."""
    fam = family_of(info["dtype"], True)
    rows, cols = logical_shape(weight)
    return (list(info["shape"])[0] == rows
            and list(info["shape"])[-1] * FAMS[fam]["scale"][1] == cols)


def mxfp8_copy_check(view: Path, view2: Path) -> dict:
    """La copia MXFP8 se corta contra la particion de la capa NVFP4 (`_tp_shard`, b12x.py:73-108): exige
    `sum(output_sizes) == filas de la copia` y columnas por rango multiplo de 32. O sea: la copia tiene que
    medir LO MISMO en unidades logicas que el peso al que sirve, aunque un empaquete 2/byte y el otro no."""
    a, b = _headers(view), _headers(view2, present_only=True)
    n_ok, problemas = 0, []
    for t, info in sorted(b.items()):
        if not t.endswith(".weight") or (t + ".weight_scale") not in b:
            continue
        if t not in a or (t + ".weight_scale") not in a:
            problemas.append(f"{t}: la copia existe y el peso NVFP4 no")
            continue
        if logical_shape(info) != logical_shape(a[t]):
            problemas.append(f"{t}: logica {logical_shape(info)} != {logical_shape(a[t])}")
        elif not scale_ok(b[t + ".weight_scale"], info) or not scale_ok(a[t + ".weight_scale"], a[t]):
            problemas.append(f"{t}: la escala no cuadra con su peso")
        else:
            n_ok += 1
    if problemas:
        raise ConversionError("copia MXFP8: " + "; ".join(problemas[:5]))
    return {"tensores_comprobados": n_ok, "pasa": True}


# --- verify: equivalencia por bytes (muestreo + cobertura exacta) ------------------------------


def _read_at(path: Path, start: int, n: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read(n)


def verify_one(src: Path, out: Path, label: str, sample: int = 1 << 20) -> dict:
    """De cada fichero reescrito: las formas de la cabecera nueva son las que el plan anuncia, los offsets
    son contiguos y el tamano cuadra, y de cada tensor se leen muestras byte a byte contra el fichero de
    origen (o contra el shard vecino, en la PLE) y contra el relleno esperado."""
    cfg = json.loads((src / "config.json").read_text())
    plan = Plan(src, cfg, TP, label)
    res = {"tensors": 0, "chunks_checked": 0, "bytes_checked": 0, "problemas": []}
    for fname in plan.rewrite:
        oheader, odata = read_header(out / fname)
        if sorted(x for x in oheader if x != "__metadata__") != sorted(x for x in plan.headers[fname][0]
                                                                      if x != "__metadata__"):
            res["problemas"].append(f"{fname}: cambian los nombres de tensor")
            continue
        off = odata
        for t in sorted((x for x in oheader if x != "__metadata__"),
                        key=lambda x: oheader[x]["data_offsets"][0]):
            oi = oheader[t]
            shape, chunks = plan.emit(t, fname, plan.headers[fname][1])
            size = sum(chunk_len(c) for c in chunks)
            if oi["dtype"] != plan.headers[fname][0][t]["dtype"]:
                res["problemas"].append(f"{fname}:{t} dtype {oi['dtype']}")
            if list(oi["shape"]) != shape:
                res["problemas"].append(f"{fname}:{t} forma {oi['shape']} != {shape}")
            if oi["data_offsets"] != [off - odata, off - odata + size]:
                res["problemas"].append(f"{fname}:{t} offsets {oi['data_offsets']} no contiguos en {off - odata}")
            res["tensors"] += 1
            at = odata + oi["data_offsets"][0]
            for i, c in enumerate(chunks):
                total = chunk_len(c)
                base_out = at + sum(chunk_len(x) for x in chunks[:i])
                if isinstance(c, bytes):
                    src_file, base_in, blob = None, 0, c
                elif len(c) == 2:
                    src_file, base_in, blob = fname, c[0], None
                else:
                    src_file, base_in, blob = c[0], c[1], None
                for shift in ({0} if total <= 2 * sample else {0, total - sample}):
                    n = min(sample, total)
                    got = _read_at(out / fname, base_out + shift, n)
                    want = (blob[shift:shift + n] if blob is not None
                            else _read_at(src / src_file, base_in + shift, n))
                    if got != want:
                        res["problemas"].append(f"{fname}:{t} trozo {i}@{shift} no coincide")
                    res["chunks_checked"] += 1
                    res["bytes_checked"] += len(got)
            off += size
        if off != (out / fname).stat().st_size:
            res["problemas"].append(f"{fname}: la cabecera anuncia {off - odata} y el fichero mide "
                                    f"{(out / fname).stat().st_size - odata}")
    return res


def verify(src: Path, out: Path, src2: Path | None, out2: Path | None) -> dict:
    rep = {"principal": verify_one(src, out, "principal")}
    if src2 and out2:
        rep["mxfp8"] = verify_one(src2, out2, "mxfp8")
    rep["pasa"] = not any(r["problemas"] for r in rep.values())
    print(json.dumps(rep, indent=2)[:4000], flush=True)
    return rep


def plan_only(src: Path, src2: Path | None) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    bad = check_source_config(text_config(cfg))
    if bad:
        raise ConversionError("config.json de origen no casa:\n  " + "\n  ".join(bad))
    reps = {"principal": Plan(src, cfg, TP, "principal").report()}
    if src2:
        reps["mxfp8"] = Plan(src2, cfg, TP, "mxfp8").report()
    print(json.dumps(reps, indent=2), flush=True)
    return reps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="convert-ursu-tp3.py")
    ap.add_argument("cmd", choices=["plan", "convert", "shapes", "verify"])
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--src2", type=Path)
    ap.add_argument("--out2", type=Path)
    ap.add_argument("--view", type=Path)
    ap.add_argument("--view2", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan_only(a.src, a.src2)
    elif a.cmd == "convert":
        convert(a.src, a.out, a.src2, a.out2)
    elif a.cmd == "shapes":
        print(json.dumps(shapes(a.view, a.view2), indent=2), flush=True)
    elif a.cmd == "verify":
        r = verify(a.src, a.out, a.src2, a.out2)
        if not r["pasa"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
