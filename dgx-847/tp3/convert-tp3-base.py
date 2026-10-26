#!/usr/bin/env python3
"""DGX-780 · vista TP=3 del checkpoint hibrido qwen38fn-mtp-nvidia-fp8-hybrid (rev 7b719225).

Con TP=3 no dividen por 3: KV 2, GDN 16:48, shared 640, vocab 248320 y el padded_vocab_size de
la PLE. Este conversor escribe una vista nueva, exacta por construccion, sin tocar la de TP=2:

  * KV 2 -> 6 (`[k0,k0,k0,k1,k1,k1]`) en k_proj/v_proj de las capas full y del MTP;
  * GDN 16:48 -> 24:72 con cabezas a cero, reparto 6/5/5 por rango; cada grupo (1 k + 3 v)
    viaja junto en in_proj_qkv, in_proj_z, in_proj_b/a, conv1d, A_log, dt_bias y out_proj;
  * shared expert 640 -> 768 con ceros (gate/up por filas, down por columnas), tambien en el MTP;
  * embed_tokens y lm_head a 248448 filas (las nuevas a cero) y `vocab_size` 248448; el
    tokenizador no cambia. `tp3_orig_vocab_size` (248320) en el config arma el parche
    QWEN_TP3_VOCAB_MASK de launch.sh, que pone -inf a los logits >= 248320;
  * escalas FP8 por bloques 128x128 movidas o duplicadas con su bloque; en el relleno, peso 0 y
    escala 1;
  * PLE: la tabla no se reescribe (sus partes son enlaces; la sirve el NVMe de cada nodo, DGX-773). Si
    padded_vocab_size no divide por 3, make_ngram_vocab_size_divisible_by sube a 384 solo en el
    config.json de la vista; el loader pide entonces 2500014 filas por parte y el checkpoint trae
    2500012, de ahi QWEN_TP3_PLE_SHARD_SKIP en launch.sh.

La conversion es copia de bytes (stdlib, sin torch): las filas y columnas se mueven enteras, no
se recuantiza nada. `verify` (torch y vllm de la imagen) comprueba la equivalencia en tensores
reales, carga los tensores convertidos en las capas reales de la imagen con tp=3, y compara los
rangos PLE de la imagen con los de build_ple_table.py y la forma de las partes con su loader.

Uso (Job `qwen38fn-tp3-convert`, python3 -I):
  convert.py convert --src DIR --out DIR
  convert.py verify  --src DIR --out DIR --json verify.json [--builder build_ple_table.py]
Jobs `qwen38fn-tp3-serve` (dgx3) y `qwen38fn-tp3-publish` (nvidia-dgx):
  convert.py serve   --out DIR --port 8080
  convert.py publish --url URL --dest DIR --sibling NOMBRE
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import types
from pathlib import Path
from unittest import mock

REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
TP = 3
BLOCK = 128  # FP8_PB_WO: bloques de 128x128
VOCAB_PAD = 64  # DEFAULT_VOCAB_PADDING_SIZE de VocabParallelEmbedding
INDEX = "model.safetensors.index.json"
BUILDER = Path(__file__).resolve().parents[1] / "qwen38-flash-next-ple-mmap/build_ple_table.py"

# C1.1: cifras del config.json de origen (text_config). Si una no casa, se aborta antes de escribir.
EXPECTED = {
    "hidden_size": 2560,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "shared_expert_intermediate_size": 640,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "num_hidden_layers": 48,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 1,
    "vocab_size": 248320,
    "tie_word_embeddings": False,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "ple_embed_dim": 2560,
    "ple_layer_ids": [2],
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "split_ngram_parts": 128,
    "ngram_vocab_size_base": 20_000_000,
    "make_ngram_vocab_size_divisible_by": 128,
    "ple_embedding_dtype": "float8_e4m3fn",
}

ITEMSIZE = {"F8_E4M3": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F64": 8}
SCALE_ONE = {"F32": struct.pack("<f", 1.0), "BF16": struct.pack("<f", 1.0)[2:]}


class ConversionError(RuntimeError):
    pass


def text_config(cfg: dict) -> dict:
    return cfg["text_config"] if "text_config" in cfg else cfg


def check_source_config(tc: dict) -> list[str]:
    """Las cifras que no casan con EXPECTED (vacio = todo bien)."""
    return [
        f"{k}: esperado {v!r}, visto {tc.get(k)!r}" for k, v in EXPECTED.items() if tc.get(k) != v
    ]


# --- PLE ---------------------------------------------------------------------------------------


def ple_view(tc: dict, rows: int, tp: int = TP) -> dict:
    """Cifras de la tabla PLE en la vista TP=3 (C1.3). `rows` son las filas del checkpoint (cabeceras), ya
    rellenas al divisor de origen: como el nuevo es multiplo de el, ceil(rows/nuevo)*nuevo es el
    padded_vocab_size de la clase de la imagen sin rehacer sus primos (C-1; C4(c) lo comprueba)."""
    div, parts = int(tc["make_ngram_vocab_size_divisible_by"]), int(tc["split_ngram_parts"])
    padded = -(-rows // div) * div
    new_div = div if padded % tp == 0 else div * tp // _gcd(div, tp)
    new = -(-rows // new_div) * new_div
    return {"checkpoint_rows": rows, "divisor": div, "padded_vocab_size": padded, "new_divisor": new_div,
            "new_padded_vocab_size": new, "pad_rows": new - rows, "rows_per_rank": new // tp,
            "checkpoint_shard_rows": -(-rows // parts), "image_shard_rows": -(-new // parts)}


# --- Mapa de cabezas --------------------------------------------------------------------------


class Layout:
    """Dimensiones de origen y de la vista TP=3, sacadas del text_config."""

    def __init__(self, tc: dict, tp: int = TP):
        self.tp = tp
        self.head_dim = int(tc["head_dim"])
        self.kv = int(tc["num_key_value_heads"])
        self.kv_new = self.kv * tp // _gcd(self.kv, tp)
        self.gk = int(tc["linear_num_key_heads"])
        self.gv = int(tc["linear_num_value_heads"])
        self.ratio = self.gv // self.gk
        # Vista TP=3 = la forma por rango de TP=2: 8 k y 24 v por rango (24:72).
        self.gk_new = self.gk // 2 * tp
        self.gv_new = self.gk_new * self.ratio
        self.kdim = int(tc["linear_key_head_dim"])
        self.vdim = int(tc["linear_value_head_dim"])
        self.shared = int(tc["shared_expert_intermediate_size"])
        self.shared_new = _round_up(self.shared, BLOCK * tp)
        self.vocab = int(tc["vocab_size"])
        self.vocab_new = _round_up(self.vocab, VOCAB_PAD * tp)
        self.groups = gdn_group_slots(self.gk, self.gk_new, tp)

    def target_config(self) -> dict:
        return {
            "num_key_value_heads": self.kv_new,
            "linear_num_key_heads": self.gk_new,
            "linear_num_value_heads": self.gv_new,
            "shared_expert_intermediate_size": self.shared_new,
            "vocab_size": self.vocab_new,
        }


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


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


# Unidad = (inicio en el origen o None para relleno, numero de filas/columnas).
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


def in_blocks(units: list) -> list:
    out = []
    for start, n in units:
        if n % BLOCK or (start is not None and start % BLOCK):
            raise ConversionError(f"unidad ({start}, {n}) no cae en bloques de {BLOCK}")
        out.append((None if start is None else start // BLOCK, n // BLOCK))
    return out


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


# --- Reglas: que tensor cambia y como ---------------------------------------------------------
# (regex, tipo, eje, unidades(L, info), es escala)
LIN = r"(?:model\.language_model|mtp)\.layers\.\d+\.linear_attn\."
RULES = (
    (r"^(?:model\.language_model|mtp)\.layers\.\d+\.self_attn\.[kv]_proj\.weight$", "kv", 0, False),
    (r"^(?:model\.language_model|mtp)\.layers\.\d+\.self_attn\.[kv]_proj\.weight_scale$", "kv", 0, True),
    (rf"^{LIN}in_proj_qkv\.weight$", "qkv", 0, False),
    (rf"^{LIN}in_proj_qkv\.weight_scale$", "qkv", 0, True),
    (rf"^{LIN}in_proj_z\.weight$", "z", 0, False),
    (rf"^{LIN}in_proj_z\.weight_scale$", "z", 0, True),
    (rf"^{LIN}in_proj_[ab]\.weight$", "ab", 0, False),
    (rf"^{LIN}conv1d\.weight$", "conv", 0, False),
    (rf"^{LIN}(?:A_log|dt_bias)$", "vhead", 0, False),
    (rf"^{LIN}out_proj\.weight$", "out", 1, False),
    (rf"^{LIN}out_proj\.weight_scale$", "out", 2, True),
    (r"\.mlp\.shared_expert\.(?:gate|up)_proj\.weight$", "shared_in", 0, False),
    (r"\.mlp\.shared_expert\.down_proj\.weight$", "shared_down", 1, False),
    (r"^(?:model\.language_model\.embed_tokens|lm_head)\.weight$", "vocab", 0, False),
)


def units_for(kind: str, L: Layout) -> list:
    if kind == "kv":
        return kv_units(L)
    if kind in ("qkv", "conv"):
        return qkv_units(L)
    if kind in ("z", "out"):
        return vhead_units(L, L.vdim)
    if kind in ("ab", "vhead"):
        return vhead_units(L, 1)
    if kind in ("shared_in", "shared_down"):
        return pad_units(L.shared, L.shared_new)
    if kind == "vocab":
        return pad_units(L.vocab, L.vocab_new)
    raise ConversionError(kind)


def expected_counts(tc: dict, fp8_scales: dict) -> dict:
    """Cuantos tensores de cada tipo tiene que tocar la conversion (derivado del config)."""
    types = tc["layer_types"]
    full = sum(t == "full_attention" for t in types)
    gdn = sum(t == "linear_attention" for t in types)
    mtp = int(tc.get("mtp_num_hidden_layers", 0))
    layers = len(types) + mtp
    out = {"kv": 2 * (full + mtp), "qkv": gdn, "z": gdn, "ab": 2 * gdn, "conv": gdn,
           "vhead": 2 * gdn, "out": gdn, "shared_in": 2 * layers, "shared_down": layers, "vocab": 2}
    for kind, n in fp8_scales.items():
        out[kind] += n
    return out


# --- safetensors en crudo (cabecera + bytes) ---------------------------------------------------


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def gather(base: int, shape: list, dtype: str, axis: int, units: list, fill: bytes) -> tuple[list, list]:
    """Reordena por bloques a lo largo de `axis` el tensor que empieza en el byte `base` del fichero.
    Devuelve (forma nueva, trozos): (inicio, fin) del fichero de origen o bytes de relleno."""
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
        raise ConversionError("relleno con el tamano del dtype")
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
    return len(c) if isinstance(c, bytes) else c[1] - c[0]


class Plan:
    """Que ficheros se reescriben, que tensores cambian y como."""

    def __init__(self, src: Path, cfg: dict, tp: int = TP):
        self.src = src
        self.cfg = cfg
        self.tc = text_config(cfg)
        self.L = Layout(self.tc, tp)
        self.weight_map: dict = json.loads((src / INDEX).read_text())["weight_map"]
        self.headers = {}
        for name in sorted(set(self.weight_map.values())):
            header, data_start = read_header(src / name)
            self.headers[name] = (header, data_start)
        self.changes: dict = {}  # tensor -> (tipo, eje, unidades, relleno)
        self.ple_shapes: dict = {}  # "ngram_embedding.shard_<i>.weight" -> forma (no se reescriben)
        fp8_scales: dict = {}
        for tensor, fname in self.weight_map.items():
            info = self.headers[fname][0][tensor]
            m = re.search(r"\.ple\.ple_embedding\.(ngram_embedding\.shard_\d+\.weight)$", tensor)
            if m:
                self.ple_shapes[m.group(1)] = info["shape"]
            for rx, kind, axis, is_scale in RULES:
                if re.search(rx, tensor):
                    units = units_for(kind, self.L)
                    if is_scale:
                        if info["dtype"] != "F32":
                            raise ConversionError(f"{tensor}: escala {info['dtype']}, se esperaba F32")
                        units = in_blocks(units)
                        fp8_scales[kind] = fp8_scales.get(kind, 0) + 1
                        fill = SCALE_ONE["F32"]
                    else:
                        fill = bytes(ITEMSIZE[info["dtype"]])
                    self._add(tensor, info, kind, axis, units, fill)
                    break
        if len(self.ple_shapes) != int(self.tc["split_ngram_parts"]):
            raise ConversionError(f"PLE: {len(self.ple_shapes)} partes y split_ngram_parts {self.tc['split_ngram_parts']}")
        self.ple = ple_view(self.tc, sum(shape[0] for shape in self.ple_shapes.values()), tp)
        self._check_counts(fp8_scales)
        self.rewrite = sorted({self.weight_map[t] for t in self.changes})

    def _add(self, tensor, info, kind, axis, units, fill):
        size = info["shape"][axis]
        real = sorted((s, n) for s, n in units if s is not None)
        if kind == "kv":  # duplicado: cada cabeza original sale `rep` veces
            ok = sorted(set(real)) == [(i * real[0][1], real[0][1]) for i in range(size // real[0][1])]
        else:  # permutacion: las unidades reales cubren el eje entero una sola vez
            ok = [s for s, _ in real] == [sum(n for _, n in real[:i]) for i in range(len(real))] and (
                sum(n for _, n in real) == size
            )
        if not ok:
            raise ConversionError(f"{tensor}: forma {info['shape']} no casa con {kind} (eje {axis})")
        self.changes[tensor] = (kind, axis, units, fill)

    def _check_counts(self, fp8_scales: dict):
        seen: dict = {}
        for kind, *_ in self.changes.values():
            seen[kind] = seen.get(kind, 0) + 1
        want = expected_counts(self.tc, fp8_scales)
        if seen != want:
            raise ConversionError(f"tensores tocados {seen} != esperados {want}")
        for tensor, (kind, *_rest) in self.changes.items():
            if tensor.endswith(".weight_scale"):
                weight = tensor[: -len("_scale")]
                if weight not in self.changes:
                    raise ConversionError(f"{tensor} sin su peso")
            elif self.headers[self.weight_map[tensor]][0][tensor]["dtype"] == "F8_E4M3":
                if tensor + "_scale" not in self.changes:
                    raise ConversionError(f"{tensor} FP8 sin su escala")

    def config(self) -> dict:
        cfg = json.loads(json.dumps(self.cfg))
        tc = text_config(cfg)
        tc.update(self.L.target_config())
        tc["make_ngram_vocab_size_divisible_by"] = self.ple["new_divisor"]
        tc["tp3_orig_vocab_size"] = self.L.vocab
        return cfg

    def files_with_experts(self) -> list[str]:
        return sorted(
            f for f in self.rewrite
            if any(".experts." in t and ".shared_expert" not in t for t in self.headers[f][0])
        )

    def output_bytes(self, fname: str) -> int:
        header, _ = self.headers[fname]
        total = 0
        for tensor, info in header.items():
            if tensor == "__metadata__":
                continue
            shape = list(info["shape"])
            if tensor in self.changes:
                kind, axis, units, _ = self.changes[tensor]
                shape[axis] = sum(n for _, n in coalesce(units))
            n = ITEMSIZE[info["dtype"]]
            for d in shape:
                n *= d
            total += n
        return total + 8 + len(header_json(header)) + 8


def header_json(header: dict) -> bytes:
    return json.dumps(header, separators=(",", ":")).encode()


def write_file(plan: Plan, fname: str, out: Path, piece: int = 64 << 20) -> tuple[str, int]:
    """Escribe `fname` en `out` (tmp + rename). Devuelve (sha256, bytes)."""
    header, data_start = plan.headers[fname]
    meta = header.get("__metadata__")
    tensors = sorted(
        (t for t in header if t != "__metadata__"), key=lambda t: header[t]["data_offsets"][0]
    )
    parts = []
    new_header: dict = {} if meta is None else {"__metadata__": meta}
    offset = 0
    for t in tensors:
        info = header[t]
        b, e = info["data_offsets"]
        shape = list(info["shape"])
        if t in plan.changes:
            kind, axis, units, fill = plan.changes[t]
            shape, chunks = gather(data_start + b, shape, info["dtype"], axis, units, fill)
        else:
            chunks = [(data_start + b, data_start + e)]
        size = sum(chunk_len(c) for c in chunks)
        if offset % ITEMSIZE[info["dtype"]]:
            raise ConversionError(f"{fname}:{t} desalineado")
        new_header[t] = {"dtype": info["dtype"], "shape": shape, "data_offsets": [offset, offset + size]}
        parts += chunks
        offset += size
    hj = header_json(new_header)
    hj += b" " * (-len(hj) % 8)
    tmp = out / (fname + ".tmp")
    digest = hashlib.sha256()
    with open(plan.src / fname, "rb") as src, open(tmp, "wb") as fh:
        mm = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for data in (struct.pack("<Q", len(hj)), hj):
                fh.write(data)
                digest.update(data)
            for c in parts:
                if isinstance(c, bytes):
                    fh.write(c)
                    digest.update(c)
                    continue
                for a in range(c[0], c[1], piece):
                    data = mm[a:min(c[1], a + piece)]
                    fh.write(data)
                    digest.update(data)
        finally:
            mm.close()
        fh.flush()
        os.fsync(fh.fileno())
    written = tmp.stat().st_size
    os.replace(tmp, out / fname)
    return digest.hexdigest(), written


def read_sums(path: Path) -> dict:
    sums = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.lstrip("*").strip()] = digest
    return sums


def convert(src: Path, out: Path, *, expected: dict | None = EXPECTED, min_free_factor: float = 1.2) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    if expected is not None:
        bad = check_source_config(text_config(cfg))
        if bad:
            raise ConversionError("config.json de origen no casa:\n  " + "\n  ".join(bad))
    plan = Plan(src, cfg)
    out.mkdir(parents=True, exist_ok=True)
    to_write = sum(plan.output_bytes(f) for f in plan.rewrite)
    free = shutil.disk_usage(out).free
    if free < min_free_factor * to_write:
        raise ConversionError(f"espacio: {free} libres < {min_free_factor} x {to_write}")
    src_sums = read_sums(src / "SHA256SUMS")
    view_files = sorted(p.name for p in src.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    missing = [f for f in view_files if f not in src_sums]
    if missing:
        raise ConversionError(f"SHA256SUMS de origen no cubre {missing[:5]}")
    sums = {}
    for fname in plan.rewrite:
        sums[fname], _ = write_file(plan, fname, out)
        print(f"reescrito {fname}", flush=True)
    cfg_bytes = (json.dumps(plan.config(), indent=2, ensure_ascii=False) + "\n").encode()
    (out / "config.json").write_bytes(cfg_bytes)
    sums["config.json"] = hashlib.sha256(cfg_bytes).hexdigest()
    links = [f for f in view_files if f not in sums]
    for f in links:
        sums[f] = src_sums[f]
    (out / "SHA256SUMS").write_text("".join(f"{sums[f]}  {f}\n" for f in sorted(sums)))
    rewritten = sum((out / f).stat().st_size for f in plan.rewrite)
    report = {
        "revision": REVISION,
        "tp": TP,
        "target": plan.L.target_config(),
        "gdn_slots": plan.L.groups,
        "ple": plan.ple,
        "rewritten_files": plan.rewrite,
        "rewritten_gib": round(rewritten / 2**30, 3),
        "rewritten_with_experts": plan.files_with_experts(),
        "links": links,
        "tensors_changed": len(plan.changes),
    }
    (out / "tp3-plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"QWEN_TP3_CONVERT reescribe {report['rewritten_gib']} GiB en {len(plan.rewrite)} ficheros "
          f"({len(links)} enlaces); PLE divisor {plan.ple['divisor']} -> {plan.ple['new_divisor']}", flush=True)
    return report


# --- verify: equivalencia (a) y cargadores reales de la imagen (b). Solo en la imagen (torch). --


def _slots_expected(L: Layout) -> list[int | None]:
    """Reparto 6/5/5 escrito de otra forma que `gdn_group_slots`, para no verificarse a si mismo."""
    per = L.gk_new // L.tp
    counts = [L.gk // L.tp + (1 if r < L.gk % L.tp else 0) for r in range(L.tp)]
    firsts = [sum(counts[:r]) for r in range(L.tp)]
    return [firsts[s // per] + s % per if s % per < counts[s // per] else None for s in range(L.gk_new)]


class _Ckpt:
    """Tensores de una vista (el origen o origen+salida) por nombre, con safe_open."""

    def __init__(self, *dirs: Path):
        self.dirs = dirs
        self.wm = json.loads((dirs[0] / INDEX).read_text())["weight_map"]

    def path(self, name: str) -> Path:
        f = self.wm[name]
        for d in reversed(self.dirs):
            if (d / f).exists():
                return d / f
        raise FileNotFoundError(f)

    def get(self, name: str):
        from safetensors import safe_open

        with safe_open(self.path(name), framework="pt") as fh:
            return fh.get_tensor(name)

    def shape(self, name: str) -> list:
        header, _ = read_header(self.path(name))
        return header[name]["shape"]


def _deq(w, s):
    """Peso FP8 por bloques 128x128 a float32 (escala [ob,1,ib,1])."""
    import torch

    s2 = s.reshape(s.shape[0], s.shape[2]).float()
    full = s2.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return w.float() * full[: w.shape[0], : w.shape[1]] if w.dtype == torch.float8_e4m3fn else w.float()


def _rows_expected(orig, units: list, fill: float = 0.0):
    """Tensor esperado a partir del original y una lista de unidades (filas)."""
    import torch

    parts = []
    for start, n in units:
        if start is None:
            parts.append(torch.full((n, *orig.shape[1:]), fill, dtype=orig.dtype))
        else:
            parts.append(orig[start:start + n])
    return torch.cat(parts, 0)


def _same(a, b) -> bool:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float8_e4m3fn:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def verify_equivalence(src: Path, out: Path) -> dict:
    """(a) capas 0 (GDN) y 3 (full) y shared de la capa 0, decuantizadas en CPU."""
    import torch

    cfg = json.loads((src / "config.json").read_text())
    L = Layout(text_config(cfg))
    o, n = _Ckpt(src), _Ckpt(src, out)
    checks = {}
    slots = _slots_expected(L)
    p = "model.language_model.layers.0.linear_attn."
    # GDN: cada hueco es su grupo original (q, k, 3 v, z, b, a, conv, A_log, dt_bias, out) o ceros.
    deq_o = _deq(o.get(p + "in_proj_qkv.weight"), o.get(p + "in_proj_qkv.weight_scale"))
    deq_n = _deq(n.get(p + "in_proj_qkv.weight"), n.get(p + "in_proj_qkv.weight_scale"))
    kd, vd, r = L.kdim, L.vdim, L.ratio
    ok = deq_n.shape[0] == 2 * L.gk_new * kd + L.gv_new * vd
    for s, g in enumerate(slots):
        for base_n, base_o, dim, cnt in ((0, 0, kd, 1), (L.gk_new * kd, L.gk * kd, kd, 1),
                                          (2 * L.gk_new * kd, 2 * L.gk * kd, vd, r)):
            new = deq_n[base_n + s * dim * cnt: base_n + (s + 1) * dim * cnt]
            ref = torch.zeros_like(new) if g is None else deq_o[base_o + g * dim * cnt: base_o + (g + 1) * dim * cnt]
            ok = ok and torch.equal(new, ref)
    checks["gdn_in_proj_qkv"] = bool(ok)
    zo = _deq(o.get(p + "in_proj_z.weight"), o.get(p + "in_proj_z.weight_scale"))
    zn = _deq(n.get(p + "in_proj_z.weight"), n.get(p + "in_proj_z.weight_scale"))
    oo = _deq(o.get(p + "out_proj.weight"), o.get(p + "out_proj.weight_scale"))
    on = _deq(n.get(p + "out_proj.weight"), n.get(p + "out_proj.weight_scale"))
    ok = zn.shape[0] == L.gv_new * vd and on.shape[1] == L.gv_new * vd
    for s, g in enumerate(slots):
        a, b = s * r * vd, (s + 1) * r * vd
        ref_z = torch.zeros_like(zn[a:b]) if g is None else zo[g * r * vd:(g + 1) * r * vd]
        ref_o = torch.zeros_like(on[:, a:b]) if g is None else oo[:, g * r * vd:(g + 1) * r * vd]
        ok = ok and torch.equal(zn[a:b], ref_z) and torch.equal(on[:, a:b], ref_o)
    checks["gdn_in_proj_z_out_proj"] = bool(ok)
    ok = True
    for name, per_head in (("in_proj_a.weight", 1), ("in_proj_b.weight", 1), ("A_log", 1), ("dt_bias", 1)):
        t_o, t_n = o.get(p + name), n.get(p + name)
        for s, g in enumerate(slots):
            new = t_n[s * r:(s + 1) * r]
            ref = torch.zeros_like(new) if g is None else t_o[g * r:(g + 1) * r]
            ok = ok and _same(new, ref)
    c_o, c_n = o.get(p + "conv1d.weight"), n.get(p + "conv1d.weight")
    for s, g in enumerate(slots):
        for base_n, base_o, dim, cnt in ((0, 0, kd, 1), (L.gk_new * kd, L.gk * kd, kd, 1),
                                          (2 * L.gk_new * kd, 2 * L.gk * kd, vd, r)):
            new = c_n[base_n + s * dim * cnt: base_n + (s + 1) * dim * cnt]
            ref = torch.zeros_like(new) if g is None else c_o[base_o + g * dim * cnt: base_o + (g + 1) * dim * cnt]
            ok = ok and _same(new, ref)
    checks["gdn_ab_alog_dt_conv"] = bool(ok)
    # Escalas por bloque: la del bloque nuevo es la del bloque original; 1 en el relleno.
    so, sn = o.get(p + "in_proj_qkv.weight_scale"), n.get(p + "in_proj_qkv.weight_scale")
    exp = _rows_expected(so, in_blocks(qkv_units(L)), 1.0)
    checks["gdn_scales_per_block"] = bool(_same(sn, exp))
    # Full (capa 3): k/v triplicadas; q y o sin cambios.
    p = "model.language_model.layers.3.self_attn."
    ok = True
    for proj in ("k_proj", "v_proj"):
        w_o, w_n = o.get(p + proj + ".weight"), n.get(p + proj + ".weight")
        s_o, s_n = o.get(p + proj + ".weight_scale"), n.get(p + proj + ".weight_scale")
        ok = ok and _same(w_n, _rows_expected(w_o, kv_units(L))) and _same(s_n, _rows_expected(s_o, in_blocks(kv_units(L))))
        d_n, d_o = _deq(w_n, s_n), _deq(w_o, s_o)
        for j in range(L.kv_new):
            hd = L.head_dim
            ok = ok and torch.equal(d_n[j * hd:(j + 1) * hd], d_o[(j * L.kv // L.kv_new) * hd:(j * L.kv // L.kv_new + 1) * hd])
    for proj in ("q_proj.weight", "q_proj.weight_scale", "o_proj.weight", "o_proj.weight_scale"):
        ok = ok and _same(n.get(p + proj), o.get(p + proj))
    checks["full_kv_triplicated"] = bool(ok)
    # Shared de la capa 0: filas/columnas reales iguales, relleno a cero.
    p = "model.language_model.layers.0.mlp.shared_expert."
    ok = True
    for proj, axis in (("gate_proj.weight", 0), ("up_proj.weight", 0), ("down_proj.weight", 1)):
        t_o, t_n = o.get(p + proj), n.get(p + proj)
        real = t_n.narrow(axis, 0, L.shared)
        pad = t_n.narrow(axis, L.shared, L.shared_new - L.shared)
        ok = ok and t_n.shape[axis] == L.shared_new and _same(real, t_o) and not pad.any().item()
    checks["shared_768"] = bool(ok)
    return {"checks": checks, "pasa": all(checks.values())}


def _install_fake_tp(rank: int, tp: int):
    """TP=3 sin GPU ni procesos: los grupos de vLLM como objetos con rango y tamano."""
    import torch
    from vllm.distributed import parallel_state as ps
    from vllm.platforms import current_platform

    class Group:
        def __init__(self, r, n):
            self.rank_in_group, self.world_size, self.rank, self.local_rank = r, n, r, r
            self.ranks = list(range(n))
            self.is_first_rank, self.is_last_rank = r == 0, r == n - 1

    for name, size in (("_TP", tp), ("_ETP", tp), ("_EP", tp), ("_WORLD", tp),
                       ("_DP", 1), ("_PP", 1), ("_DCP", 1), ("_PCP", 1)):
        setattr(ps, name, Group(rank if size == tp else 0, size))
    # Sin GPU la plataforma es Unspecified: sin dispositivo ni kernels de GEMM. Aqui solo se
    # prueban create_weights y los weight_loader, nunca un forward.
    current_platform.current_device = lambda: torch.device("cpu")
    import vllm.model_executor.layers.quantization.modelopt as modelopt

    modelopt.select_linear_kernel = lambda *a, **k: None
    modelopt.expose_input_quant_key = lambda *a, **k: None


def _vllm_config(tc_obj, quant_config):
    import types

    import torch
    from vllm.config import DeviceConfig, VllmConfig

    vc = VllmConfig(device_config=DeviceConfig(device="cpu"))
    mc = types.SimpleNamespace(dtype=torch.bfloat16, hf_text_config=tc_obj, hf_config=tc_obj,
                               multimodal_config=None, uses_mrope=True, head_dtype=None)
    object.__setattr__(vc, "model_config", mc)
    object.__setattr__(vc, "quant_config", quant_config)
    return vc


def verify_loaders(src: Path, out: Path, tp: int = TP) -> dict:
    """(b) Para cada rango, las capas reales de la imagen con tp=3 cargan los tensores convertidos
    con su weight_loader: lo real igual al original, el relleno a cero."""
    import torch
    from torch import nn

    o, n = _Ckpt(src), _Ckpt(src, out)
    cfg_new = json.loads((out / "config.json").read_text())
    tc_new = text_config(cfg_new)
    L = Layout(text_config(json.loads((src / "config.json").read_text())), tp)
    slots = _slots_expected(L)
    results: dict = {}
    torch.set_default_dtype(torch.bfloat16)
    for rank in range(tp):
        _install_fake_tp(rank, tp)
        import vllm.models.qwen4_exp.nvidia.model as qmodel
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        from vllm.model_executor.model_loader.utils import configure_quant_config
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
        from vllm.model_executor.models.qwen3_next import Qwen3NextMLP
        from vllm.model_executor.models.utils import AutoWeightsLoader
        from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

        tconf = Qwen4ExpTextConfig(**tc_new)
        qc = ModelOptMixedPrecisionConfig.from_config(cfg_new["quantization_config"])
        configure_quant_config(qc, qmodel.Qwen4ExpForConditionalGeneration)
        vc = _vllm_config(tconf, qc)
        checks: dict = {}
        hd, kd, vd, r = L.head_dim, L.kdim, L.vdim, L.ratio

        with set_current_vllm_config(vc):
            # GDN de la capa 0 (prefijo vLLM: language_model.model.layers.0.linear_attn).
            gdn = QwenGatedDeltaNetAttention(tconf, vllm_config=vc, gqa_interleaved_layout=False,
                                             prefix="language_model.model.layers.0.linear_attn")
            p = "model.language_model.layers.0.linear_attn."
            names = ["in_proj_qkv.weight", "in_proj_qkv.weight_scale", "in_proj_z.weight",
                     "in_proj_z.weight_scale", "in_proj_b.weight", "in_proj_a.weight", "conv1d.weight",
                     "A_log", "dt_bias", "norm.weight", "out_proj.weight", "out_proj.weight_scale"]
            holder = nn.Module()
            holder.add_module("linear_attn", gdn)
            AutoWeightsLoader(holder).load_weights(
                (("linear_attn." + nm, n.get(p + nm)) for nm in names),
                mapper=_stacked_mapper(Qwen3_5Model.hf_to_vllm_mapper),
            )
            mine = slots[rank * (L.gk_new // tp):(rank + 1) * (L.gk_new // tp)]
            q_o = o.get(p + "in_proj_qkv.weight")
            qs_o = o.get(p + "in_proj_qkv.weight_scale")
            z_o, zs_o = o.get(p + "in_proj_z.weight"), o.get(p + "in_proj_z.weight_scale")

            def rows(per_slot_units):
                return [u for g in mine for u in per_slot_units(g)]

            q_u = rows(lambda g: [(None if g is None else g * kd, kd)])
            k_u = rows(lambda g: [(None if g is None else L.gk * kd + g * kd, kd)])
            v_u = rows(lambda g: [(None if g is None else 2 * L.gk * kd + (g * r + j) * vd, vd) for j in range(r)])
            z_u = rows(lambda g: [(None if g is None else (g * r + j) * vd, vd) for j in range(r)])
            w_exp = torch.cat([_rows_expected(q_o, q_u + k_u + v_u), _rows_expected(z_o, z_u)], 0)
            s_exp = torch.cat([_rows_expected(qs_o, in_blocks(q_u + k_u + v_u), 1.0),
                               _rows_expected(zs_o, in_blocks(z_u), 1.0)], 0)
            checks["gdn.in_proj_qkvz"] = _same(gdn.in_proj_qkvz.weight.data, w_exp) and _same(
                gdn.in_proj_qkvz.weight_scale.data, s_exp)
            hv = rows(lambda g: [(None if g is None else g * r + j, 1) for j in range(r)])
            ba = torch.cat([_rows_expected(o.get(p + "in_proj_b.weight"), hv),
                            _rows_expected(o.get(p + "in_proj_a.weight"), hv)], 0)
            checks["gdn.in_proj_ba"] = _same(gdn.in_proj_ba.weight.data, ba)
            conv = _rows_expected(o.get(p + "conv1d.weight"), q_u + k_u + v_u)
            checks["gdn.conv1d"] = _same(gdn.conv1d.weight.data, conv)
            checks["gdn.A_log"] = torch.equal(gdn.A_log.data, _rows_expected(o.get(p + "A_log"), hv).float())
            checks["gdn.dt_bias"] = _same(gdn.dt_bias.data, _rows_expected(o.get(p + "dt_bias"), hv))
            oc = [u for u in z_u]
            out_w = _rows_expected(o.get(p + "out_proj.weight").t().contiguous(), oc).t()
            out_s = _rows_expected(o.get(p + "out_proj.weight_scale").permute(2, 1, 0, 3).contiguous(),
                                   in_blocks(oc), 1.0).permute(2, 1, 0, 3)
            checks["gdn.out_proj"] = _same(gdn.out_proj.weight.data, out_w.contiguous()) and _same(
                gdn.out_proj.weight_scale.data, out_s.contiguous())
            del gdn, holder
            # Full de la capa 3: los pesos de Qwen4ExpQSAAttention (qsa.py de la imagen) con sus mismos
            # constructores. La clase entera exige FlashAttention en CUDA; el resto del __init__ no
            # tiene pesos. Mismos argumentos que el overlay de ursucipian (solo cambia el dtype de KV).
            p = "model.language_model.layers.3.self_attn."
            pre = "language_model.model.layers.3.self_attn"
            att = nn.Module()
            att.qkv_proj = QKVParallelLinear(tconf.hidden_size, tconf.head_dim, tconf.num_attention_heads * 2,
                                             tconf.num_key_value_heads, bias=False,
                                             quant_config=qmodel.without_modelopt_fp4(qc), prefix=f"{pre}.qkv_proj")
            att.o_proj = RowParallelLinear(tconf.num_attention_heads * tconf.head_dim, tconf.hidden_size,
                                           bias=False, quant_config=qc, prefix=f"{pre}.o_proj")
            indexer = nn.Module()
            indexer.index_qk_proj = ReplicatedLinear(
                tconf.hidden_size, (tconf.indexer_n_heads + tconf.indexer_kv_heads) * tconf.indexer_head_dim,
                bias=False, quant_config=qc, prefix=f"{pre}.indexer.index_qk_proj")
            att.indexer = indexer
            holder = nn.Module()
            holder.add_module("self_attn", att)
            names = ["q_proj.weight", "q_proj.weight_scale", "k_proj.weight", "k_proj.weight_scale",
                     "v_proj.weight", "v_proj.weight_scale", "o_proj.weight", "o_proj.weight_scale",
                     "indexer.index_qk_proj.weight"]
            AutoWeightsLoader(holder).load_weights(
                (("self_attn." + nm, n.get(p + nm)) for nm in names),
                mapper=_stacked_mapper(Qwen3_5Model.hf_to_vllm_mapper),
            )
            nq = tconf.num_attention_heads * 2 // tp * hd
            q_o, qs_o = o.get(p + "q_proj.weight"), o.get(p + "q_proj.weight_scale")
            per_kv = L.kv_new // tp
            kv_u = [((j * L.kv // L.kv_new) * hd, hd) for j in range(rank * per_kv, (rank + 1) * per_kv)]
            w_exp = torch.cat([q_o[rank * nq:(rank + 1) * nq], _rows_expected(o.get(p + "k_proj.weight"), kv_u),
                               _rows_expected(o.get(p + "v_proj.weight"), kv_u)], 0)
            s_exp = torch.cat([qs_o[rank * nq // BLOCK:(rank + 1) * nq // BLOCK],
                               _rows_expected(o.get(p + "k_proj.weight_scale"), in_blocks(kv_u)),
                               _rows_expected(o.get(p + "v_proj.weight_scale"), in_blocks(kv_u))], 0)
            checks["full.qkv_proj"] = _same(att.qkv_proj.weight.data, w_exp) and _same(
                att.qkv_proj.weight_scale.data, s_exp)
            ow, os_ = o.get(p + "o_proj.weight"), o.get(p + "o_proj.weight_scale")
            no = tconf.num_attention_heads * hd // tp
            checks["full.o_proj"] = _same(att.o_proj.weight.data, ow[:, rank * no:(rank + 1) * no].contiguous()) and _same(
                att.o_proj.weight_scale.data, os_[:, :, rank * no // BLOCK:(rank + 1) * no // BLOCK].contiguous())
            checks["full.indexer_replicated"] = _same(att.indexer.index_qk_proj.weight.data,
                                                      o.get(p + "indexer.index_qk_proj.weight"))
            del att, holder
            # Shared de la capa 0.
            p = "model.language_model.layers.0.mlp.shared_expert."
            mlp = Qwen3NextMLP(hidden_size=tconf.hidden_size, intermediate_size=tconf.shared_expert_intermediate_size,
                               hidden_act="silu", quant_config=qc, reduce_results=False,
                               prefix="language_model.model.layers.0.mlp.shared_expert")
            holder = nn.Module()
            holder.add_module("shared_expert", mlp)
            AutoWeightsLoader(holder).load_weights(
                (("shared_expert." + nm, n.get(p + nm)) for nm in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")),
                mapper=_stacked_mapper(Qwen3_5Model.hf_to_vllm_mapper),
            )
            per = L.shared_new // tp
            u = [(rank * per, min(per, max(0, L.shared - rank * per))) if rank * per < L.shared else (None, 0),
                 (None, per - min(per, max(0, L.shared - rank * per)))]
            gate = _rows_expected(o.get(p + "gate_proj.weight"), u)
            up = _rows_expected(o.get(p + "up_proj.weight"), u)
            down = _rows_expected(o.get(p + "down_proj.weight").t().contiguous(), u).t().contiguous()
            checks["shared.gate_up_proj"] = _same(mlp.gate_up_proj.weight.data, torch.cat([gate, up], 0))
            checks["shared.down_proj"] = _same(mlp.down_proj.weight.data, down)
            del mlp, holder
            # VocabParallelEmbedding / ParallelLMHead.
            per = L.vocab_new // tp
            u = [(rank * per, min(per, max(0, L.vocab - rank * per))), (None, per - min(per, max(0, L.vocab - rank * per)))]
            for label, layer, name in (
                ("embed_tokens", VocabParallelEmbedding(L.vocab_new, tconf.hidden_size), "model.language_model.embed_tokens.weight"),
                ("lm_head", ParallelLMHead(L.vocab_new, tconf.hidden_size, prefix="language_model.lm_head"), "lm_head.weight"),
            ):
                layer.weight.weight_loader(layer.weight, n.get(name))
                checks[label] = _same(layer.weight.data, _rows_expected(o.get(name), u))
                del layer
        results[f"rank{rank}"] = {"checks": checks, "pasa": all(checks.values())}
    return {"ranks": results, "pasa": all(v["pasa"] for v in results.values())}


def _stacked_mapper(base):
    """El mapper de Qwen3.5 (in_proj_qkv -> in_proj_qkvz (0,1,2), ...) mas q/k/v -> qkv_proj y
    gate/up -> gate_up_proj, que en el modelo hace el load_weights de cada capa."""
    from vllm.model_executor.models.utils import WeightsMapper

    return base | WeightsMapper(orig_to_new_stacked={
        ".q_proj.": (".qkv_proj.", "q"),
        ".k_proj.": (".qkv_proj.", "k"),
        ".v_proj.": (".qkv_proj.", "v"),
        ".gate_proj.": (".gate_up_proj.", 0),
        ".up_proj.": (".gate_up_proj.", 1),
    })


def link_view(src: Path, out: Path, dest: Path) -> Path:
    """La vista TP=3 entera en `dest`: lo reescrito de `out`, lo demas de `src` (enlaces absolutos)."""
    dest.mkdir(parents=True)
    for f in src.iterdir():
        if f.name != "SHA256SUMS":
            (dest / f.name).symlink_to((out / f.name if (out / f.name).exists() else f).resolve())
    return dest


def ngram_load_weights(path: Path):
    """`Qwen4ExpNGramEmbedding.load_weights` del fichero `path` (el de la imagen, con o sin
    QWEN_TP3_PLE_SHARD_SKIP) para correrlo sin torch: hasta su weight_loader solo mira formas. El parche
    puede usar `os` y `self.ngram_embedding.tp_size`, nada mas del modulo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpNGramEmbedding")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    ns: dict = {"os": os}
    exec(compile(ast.Module([fn], []), str(path), "exec"), ns)
    return ns["load_weights"]


def ple_shard_check(load_weights, tc: dict, shapes: dict, org_vocab_size: int, tp: int = TP) -> dict:
    """(d) El chequeo de forma de ngram_embedding.py:908-921 con las cabeceras de la vista (ni bytes ni
    memoria), como en el pod: QWEN_PLE_MMAP_DIR puesto. Sin el parche, en TP=3 falla en la parte 0."""
    heads = (int(tc["ngram_size"]) - 1) * int(tc["heads_per_ngram"])
    emb = types.SimpleNamespace(org_vocab_size=org_vocab_size, embedding_dim=int(tc["ple_embed_dim"]) // heads,
                                tp_size=tp, weight=types.SimpleNamespace(weight_loader=lambda *a, **k: None))
    me = types.SimpleNamespace(ngram_embedding=emb, split_ngram_parts=int(tc["split_ngram_parts"]),
                               layer_multipliers=None, ngram_heads_offsets=None, ngram_heads_vocab_sizes=None)
    weights = [(name, types.SimpleNamespace(shape=tuple(shape))) for name, shape in shapes.items()]
    with mock.patch.dict(os.environ, {"QWEN_PLE_MMAP_DIR": os.environ.get("QWEN_PLE_MMAP_DIR", "/ple")}):
        try:
            loaded = load_weights(me, weights)
        except ValueError as exc:
            return {"pasa": False, "error": str(exc)}
    # C-7: el parche se salta la forma y el weight_loader, no la marca de tabla cargada
    return {"pasa": "ngram_embedding.weight" in loaded, "loaded": sorted(loaded)}


def verify_ple(src: Path, out: Path, builder: Path, tp: int = TP) -> dict:
    """(c) Los rangos PLE de la imagen, con sus funciones estaticas y sin instanciar la tabla (un rango son
    ~15,9 GiB y el Job tiene 16Gi), iguales a los de build_ple_table.plan: es lo que ple_mmap.py:69-79
    compara con ready.json al arrancar. (d) La forma de las partes con el loader de la imagen."""
    import importlib.util
    import tempfile

    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, pad_vocab_size

    tc = text_config(json.loads((out / "config.json").read_text()))
    plan = Plan(src, json.loads((src / "config.json").read_text()), tp)
    heads = (int(tc["ngram_size"]) - 1) * int(tc["heads_per_ngram"])
    # una sola capa PLE: model.py:199-202 le da ple_dense_layer_id 0
    _, _, total = ng.Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=int(tc["ngram_vocab_size_base"]), ngram_heads=heads, ple_dense_layer_id=0)
    div = int(tc["make_ngram_vocab_size_divisible_by"])
    padded = (total + div - 1) // div * div  # ngram_embedding.py:693-694
    pad = pad_vocab_size(padded, div)
    checks = {"padded_vocab_size": padded == plan.ple["new_padded_vocab_size"]}
    spec = importlib.util.spec_from_file_location("build_ple_table", builder)
    bpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpt)
    with tempfile.TemporaryDirectory() as tmp:
        view = link_view(src, out, Path(tmp) / "view")
        for r in range(tp):
            idx = VocabParallelEmbedding._get_indices(pad, pad, padded, padded, r, tp)
            got = (idx.org_vocab_start_index, idx.org_vocab_end_index)
            try:
                rows = bpt.plan(view, tp, r)
                checks[f"rank{r}"] = got == (rows["row_start"], rows["row_end"])
            except SystemExit as exc:  # fail() de build_ple_table
                print(exc, flush=True)
                checks[f"rank{r}"] = False
            print(f"ple rank{r}: filas {got} de {padded} (primos {total})", flush=True)
    shapes = ple_shard_check(ngram_load_weights(Path(ng.__file__)), tc, plan.ple_shapes, padded, tp)
    return {"rangos": {"checks": checks, "pasa": all(checks.values())}, "formas": shapes}


def verify(src: Path, out: Path, json_path: Path | None, builder: Path = BUILDER) -> dict:
    report = {"revision": REVISION, "equivalencia": verify_equivalence(src, out)}
    report["verify_loaders"] = verify_loaders(src, out)
    report["ple"] = verify_ple(src, out, builder)
    report["pasa"] = bool(report["equivalencia"]["pasa"] and report["verify_loaders"]["pasa"]
                          and report["ple"]["rangos"]["pasa"] and report["ple"]["formas"]["pasa"])
    text = json.dumps(report, indent=2) + "\n"
    if json_path:
        json_path.write_text(text)
    print(text, flush=True)
    return report


# --- serve y publish: de dgx3 a dgx1 por la red del CNI (Jobs serve y publish, C10) ----------

MARK = f".built-tp3-{REVISION[:8]}"
FIN = "/fin"


def serve(out: Path, port: int) -> None:
    """HTTP de solo lectura (GET/HEAD) sobre la vista de dgx3; GET /fin lo apaga y el Job acaba."""
    import functools
    import http.server
    import threading

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == FIN:
                self.send_response(200)
                self.end_headers()
                threading.Thread(target=srv.shutdown).start()
                return
            super().do_GET()

    srv = http.server.ThreadingHTTPServer(("", port), functools.partial(Handler, directory=str(out)))
    print(f"QWEN_TP3_SERVE {out} en :{port}", flush=True)
    with srv:
        srv.serve_forever()


def publish(url: str, dest: Path, sibling: str, *, poll: float = 60.0, min_free_factor: float = 1.2) -> None:
    """Trae de `serve` lo reescrito, enlaza lo demas a la vista hermana (relativo), `sha256sum -c` de la
    vista entera y renombra de forma atomica. Espera a la marca de `convert` y a su verify.json PASA."""
    import subprocess
    import time
    import urllib.error
    import urllib.request

    def get(name: str) -> bytes:
        with urllib.request.urlopen(f"{url}/{name}", timeout=60) as r:
            return r.read()

    if dest.exists():
        print(f"QWEN_TP3_PUBLISH ya publicado: {dest}", flush=True)
        return
    while True:
        try:
            get(MARK)
            break
        except (urllib.error.URLError, OSError) as exc:
            print(f"QWEN_TP3_PUBLISH esperando {MARK}: {exc}", flush=True)
            time.sleep(poll)
    if not json.loads(get("verify.json")).get("pasa"):
        raise ConversionError("verify.json de dgx3 no PASA")
    plan = json.loads(get("tp3-plan.json"))
    sums = get("SHA256SUMS")
    names = plan["rewritten_files"] + ["config.json"]
    need = 0
    for f in names:
        with urllib.request.urlopen(urllib.request.Request(f"{url}/{f}", method="HEAD"), timeout=60) as r:
            need += int(r.headers["Content-Length"])
    free = shutil.disk_usage(dest.parent).free
    if free < min_free_factor * need:
        raise ConversionError(f"espacio en {dest.parent}: {free} libres < {min_free_factor} x {need}")
    tmp = dest.parent / f".{dest.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for f in names + ["tp3-plan.json", "verify.json"]:
        with urllib.request.urlopen(f"{url}/{f}", timeout=600) as r, open(tmp / f, "wb") as fh:
            shutil.copyfileobj(r, fh, 64 << 20)
    (tmp / "SHA256SUMS").write_bytes(sums)
    for f in plan["links"]:
        os.symlink(f"../{sibling}/{f}", tmp / f)
    subprocess.run(["sha256sum", "--check", "--quiet", "--strict", "SHA256SUMS"], cwd=tmp, check=True)
    os.rename(tmp, dest)
    print(f"QWEN_TP3_PUBLISH {dest}: {len(names)} ficheros traidos ({need / 2**30:.3f} GiB), "
          f"{len(plan['links'])} enlaces a ../{sibling}", flush=True)
    get(FIN.lstrip("/"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("convert", "verify", "plan", "serve", "publish"))
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--url")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--sibling")
    ap.add_argument("--builder", type=Path, default=BUILDER)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "serve":
            serve(args.out, args.port)
            return 0
        if args.cmd == "publish":
            publish(args.url, args.dest, args.sibling)
            return 0
        if args.cmd == "plan":
            cfg = json.loads((args.src / "config.json").read_text())
            print(json.dumps({"ple": Plan(args.src, cfg).ple, "bad": check_source_config(text_config(cfg))}, indent=2))
            return 0
        if args.cmd == "convert":
            convert(args.src, args.out)
            return 0
        return 0 if verify(args.src, args.out, args.json, args.builder)["pasa"] else 1
    except ConversionError as exc:
        print(f"QWEN_TP3_CONVERT ABORTA: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
