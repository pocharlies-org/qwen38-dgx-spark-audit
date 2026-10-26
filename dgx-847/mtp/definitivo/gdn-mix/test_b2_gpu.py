#!/usr/bin/env python3
"""DGX-852 B2 · prueba de kernel del prefill b12x con estado SSM BF16 (en GPU).

Se corre DENTRO de la imagen de ursuciprian en un Spark. Esta sesion NO puede
correrla (no toca clúster ni Sparks): lo lanza una persona con GPU.

1. Copia el paquete `b12x` instalado a un tmp y aplica `patch_gdn_mix.py` en modo
   B2 sobre la COPIA, con prioridad en sys.path. No toca el instalado.
2. Con entradas aleatorias y formas reales (K=V=128; cabezas locales de TP=2 y
   TP=1) corre el prefill b12x con el estado en BF16 y en FP32, y los compara
   contra el oraculo `b12x.sequence.gdn_prefill.reference.prefill_gdn`.

La comparacion que prueba B2 es b12x-BF16 vs oraculo-con-el-MISMO-estado-BF16:
ambos arrancan de la misma base redondeada, asi que solo difieren en la
conversion del kernel (cargar BF16 -> operar FP32 -> guardar BF16). Si el kernel
tratara mal el BF16, ahi se ve. FP32-vs-oraculo-FP32 sigue siendo el ancla
absoluta de la correccion del kernel.

Se comprueban salida, estado final y checkpoints, y se INTENTA la ruta de
transferencia empaquetada (algorithm=chunk_parallel: _PackTransferKernel /
_BoundaryKernel / _CommitKernel), que es donde el pool FP32 convive con la base
BF16. El default de vLLM es sequential; chunk_parallel solo se activa por
override, asi que si la validacion de config o la memoria compartida lo rechaza
se informa como SALTO, no como fallo.

Tolerancias: relativas a la magnitud del resultado de referencia. BF16 redondea
a ~2^-9 (0.2 %) por valor; los limites son holgados pero excluyen un error de
tipo (tratar BF16 como FP32 o al reves da errores de anos-magnitud o NaN).

Uso (dentro del contenedor, en el Spark):
    python3 -I test_b2_gpu.py
    python3 -I test_b2_gpu.py --key-heads 16      # solo cabezas locales TP=1
    python3 -I test_b2_gpu.py --no-chunk          # solo la ruta sequential
    B12X_FORK_SRC=/ruta/dist-packages python3 -I test_b2_gpu.py   # si find_spec falla

Sale 0 con todo OK; 1 si hay algun FALLO.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HEAD_DIM = 128
# ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE: linear_num_key_heads=16,
# linear_num_value_heads=48 (value = 3*key), head_dim=128. Por rango de TP:
TP_RANGES = [(8, 24, "TP=2"), (16, 48, "TP=1")]

FAILURES: list[str] = []
SKIPS: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    FAILURES.append(msg)
    log(f"  FALLO {msg}")


def ok(msg: str) -> None:
    log(f"  ok    {msg}")


def skip(msg: str) -> None:
    SKIPS.append(msg)
    log(f"  SALTO {msg}")


# --- copia parcheada ---------------------------------------------------------
def installed_dist_packages() -> Path:
    explicit = os.environ.get("B12X_FORK_SRC")
    if explicit:
        return Path(explicit)
    spec = importlib.util.find_spec("b12x")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("no localizo b12x instalado; pasa B12X_FORK_SRC=<dist-packages>")
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def patched_root(here: Path) -> Path:
    src = installed_dist_packages()
    root = Path(tempfile.mkdtemp(prefix="b2gpu-")) / "dist-packages"
    (root / "vllm").mkdir(parents=True)  # el parcheador exige vllm/ y b12x/ juntos
    shutil.copytree(src / "b12x", root / "b12x")
    patcher = here / "patch_gdn_mix.py"
    if not patcher.is_file():
        raise SystemExit(f"falta el parcheador: {patcher}")
    env = dict(os.environ)
    env["QWEN_GDN_MIX"] = "B2"
    r = subprocess.run([sys.executable, "-I", str(patcher), "--root", str(root)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"el parcheador B2 fallo sobre la copia (exit {r.returncode}):\n"
                         f"{r.stdout}{r.stderr}")
    log(f"b12x copiado y parcheado (B2) en {root}")
    return root


def load_patched(root: Path):
    for name in [m for m in sys.modules if m == "b12x" or m.startswith("b12x.")]:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    api = importlib.import_module("b12x.sequence.gdn_prefill.api")
    ref = importlib.import_module("b12x.sequence.gdn_prefill.reference")
    tuning = importlib.import_module("b12x.sequence.gdn_prefill._tuning")
    loaded = Path(api.__file__).resolve()
    if not str(loaded).startswith(str(root.resolve())):
        raise SystemExit(f"se cargo b12x de {loaded}, no de la copia {root}")
    if "QWEN_GDN_MIX_PREFCAPS" not in (root / "b12x/sequence/gdn_prefill/_impl.py").read_text():
        raise SystemExit("la copia cargada no lleva la marca B2 en _impl.py")
    log(f"b12x parcheado cargado desde {loaded.parent}")
    return api, ref, tuning


# --- entradas ----------------------------------------------------------------
def make_inputs(key_heads, value_heads, tokens, num_seqs, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype).to(device)

    q = rand(tokens, key_heads, HEAD_DIM)
    k = rand(tokens, key_heads, HEAD_DIM)
    v = rand(tokens, value_heads, HEAD_DIM)
    a = rand(tokens, value_heads, scale=0.5)
    b = rand(tokens, value_heads, scale=2.0)
    A_log = rand(value_heads, dtype=torch.float32, scale=0.5)
    dt_bias = rand(value_heads, dtype=torch.float32, scale=0.2)
    base = tokens // num_seqs
    lens = [base + (1 if i < tokens - base * num_seqs else 0) for i in range(num_seqs)]
    cu, acc = [0], 0
    for length in lens:
        acc += length
        cu.append(acc)
    slots = 3 * num_seqs + 1  # 0=null; 1..N iniciales, N+1..2N finales, 2N+1..3N checkpoints
    return dict(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
        initial=torch.arange(1, num_seqs + 1, dtype=torch.int32, device=device),
        final=torch.arange(num_seqs + 1, 2 * num_seqs + 1, dtype=torch.int32, device=device),
        ckpt=torch.arange(2 * num_seqs + 1, 3 * num_seqs + 1, dtype=torch.int32, device=device),
        ckpt_offsets=torch.tensor([max(1, length // 2) for length in lens],
                                  dtype=torch.int32, device=device),
        num_seqs=torch.tensor([num_seqs], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([tokens], dtype=torch.int32, device=device),
        slots=slots, tokens=tokens, n_seqs=num_seqs, value_heads=value_heads,
    )


def make_state(inp, dtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 991)
    state = torch.randn(inp["slots"], inp["value_heads"], HEAD_DIM, HEAD_DIM,
                        generator=g, dtype=torch.float32) * 0.3
    return state.to(dtype).to(device)


def bind_kwargs(inp, state, scratch, output):
    return dict(
        scratch=scratch, q=inp["q"], k=inp["k"], v=inp["v"], a=inp["a"], b=inp["b"],
        A_log=inp["A_log"], dt_bias=inp["dt_bias"], recurrent_state=state,
        cu_seqlens=inp["cu_seqlens"], initial_state_indices=inp["initial"],
        final_state_indices=inp["final"], checkpoint_state_indices=inp["ckpt"],
        checkpoint_offsets=inp["ckpt_offsets"], num_seqs=inp["num_seqs"],
        num_tokens=inp["num_tokens"], output=output,
    )


def run_b12x(api, inp, state, *, caps_kwargs, override):
    """Prepara, bindea y corre el prefill b12x; `state` se escribe in-place.

    Se prepara con `prepare_default` y un prepare_call que bindea y corre, igual
    que vLLM en `_b12x_gdn_prefill_call`, y luego se mide con `api.bind`/`api.run`
    sobre un scratch nuevo. Devuelve (output, state).
    """
    from b12x.preparation import PreparedCall, prepare_default

    device = state.device
    output = torch.zeros_like(inp["v"])
    caps = api.Caps(device=device, state_dtype=state.dtype, **caps_kwargs)
    plan = api.plan(
        caps,
        invocation=api.invocation_from_tensors(
            A_log=inp["A_log"], dt_bias=inp["dt_bias"],
            initial_state_indices=torch.empty(0, dtype=torch.int32, device=device),
        ),
        override=override,
    )
    specs = tuple(plan.scratch_specs())
    if len(specs) != 1:
        raise RuntimeError(f"b12x GDN prefill espera 1 scratch, hay {len(specs)}")

    def new_scratch():
        return torch.empty(specs[0].shape, dtype=specs[0].dtype, device=device)

    def prepare_call(pstate):
        scratch = new_scratch()
        binding = pstate.bind(**bind_kwargs(inp, state, scratch, output))
        return PreparedCall(
            run=lambda: pstate.run(binding, max_live_tokens=inp["tokens"],
                                   max_live_seqs=inp["n_seqs"]),
            produce=lambda: None, reset=lambda: None, owners=(scratch,),
        )

    prepare_default(plan.request(name="b2gpu", prepare_call=prepare_call,
                                 benchmark_call=prepare_call))
    binding = api.bind(plan, **bind_kwargs(inp, state, new_scratch(), output))
    out = api.run(binding, max_live_tokens=inp["tokens"], max_live_seqs=inp["n_seqs"])
    torch.cuda.synchronize()
    return out, state


def run_reference(ref, inp, state):
    """Oraculo secuencial; escribe final y checkpoint en `state` in-place."""
    out = ref.prefill_gdn(
        inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"], inp["dt_bias"],
        state, inp["cu_seqlens"], inp["initial"], inp["final"], inp["ckpt"],
        inp["ckpt_offsets"], inp["num_seqs"], inp["num_tokens"],
        null_state_index=0, output=torch.zeros_like(inp["v"]),
    )
    torch.cuda.synchronize()
    return out, state


# --- comparaciones -----------------------------------------------------------
def rel(a, b):
    """Error maximo de `a` contra `b`, relativo a la magnitud de `b`."""
    a, b = a.float(), b.float()
    diff = (a - b).abs().max().item()
    return diff, diff / max(b.abs().max().item(), 1e-6)


def check(name, a, b, limit, *, exact_dtype=None):
    if exact_dtype is not None and a.dtype != exact_dtype:
        fail(f"{name}: dtype {a.dtype}, esperado {exact_dtype}")
        return
    if not torch.isfinite(a.float()).all():
        fail(f"{name}: NaN/inf en el resultado")
        return
    diff, r = rel(a, b)
    if r > limit:
        fail(f"{name}: rel={r:.4%} (abs={diff:.5f}) por encima de {limit:.2%}")
    else:
        ok(f"{name}: rel={r:.4%} abs={diff:.5f}")


def bf16_round(x):
    return x.to(torch.bfloat16).to(torch.float32)


def chunk_override(tuning, caps_kwargs, segment_tokens=128):
    segments = -(-caps_kwargs["max_tokens"] // segment_tokens) + caps_kwargs["max_seqs"]
    return tuning.GdnPrefillConfig(
        backend=tuning.BACKEND, v_split=64, k_split=1, stages=2,
        window_tiles=tuning.tiles_capacity(caps_kwargs["max_tokens"], segments),
        algorithm="chunk_parallel", segment_tokens=segment_tokens,
    )


def check_chunk(api, ref, tuning, caps_base, inp, tag, device):
    """Ruta empaquetada: boundary (lee la base) + commit (escribe base y ckpt)."""
    caps_kwargs = dict(caps_base, max_tokens=inp["tokens"], max_seqs=inp["num_seqs"])
    try:
        ov = chunk_override(tuning, caps_kwargs)
    except Exception as exc:  # noqa: BLE001
        skip(f"{tag} chunk_parallel: config rechazada: {exc}")
        return
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        state = make_state(inp, dtype, device, seed=13)
        try:
            results[dtype] = run_b12x(api, inp, state, caps_kwargs=caps_kwargs, override=ov)
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if any(w in text for w in ("shared memory", "window_tiles", "unsupported",
                                       "must equal", "at most", "requires")):
                skip(f"{tag} chunk_parallel no es configurable aqui: {text[:180]}")
            else:
                fail(f"{tag} chunk_parallel {dtype}: {text[:220]}")
            return
    out_ref, st_ref = run_reference(ref, inp, make_state(inp, torch.float32, device, seed=13))
    out32, st32 = results[torch.float32]
    out16, st16 = results[torch.bfloat16]
    log(f"  [{tag} chunk_parallel]")
    check("chunk fp32 salida vs oraculo", out32, out_ref, 0.02)
    check("chunk fp32 final vs oraculo", st32[inp["final"]], st_ref[inp["final"]], 0.02)
    check("chunk bf16 salida vs oraculo", out16, out_ref, 0.03)
    check("chunk bf16 final vs bf16(oraculo)", st16[inp["final"]],
          bf16_round(st_ref[inp["final"]]), 0.03, exact_dtype=torch.bfloat16)
    check("chunk bf16 ckpt vs bf16(oraculo)", st16[inp["ckpt"]],
          bf16_round(st_ref[inp["ckpt"]]), 0.03, exact_dtype=torch.bfloat16)


def check_case(api, ref, tuning, key_heads, value_heads, tp, device, do_chunk):
    tokens, num_seqs = 256, 2
    inp = make_inputs(key_heads, value_heads, tokens, num_seqs, device, seed=7)
    caps = dict(max_tokens=tokens, max_seqs=num_seqs, max_state_slots=inp["slots"],
                key_heads=key_heads, value_heads=value_heads,
                checkpoint_export=True, null_state_index=0)
    log(f"[{tp}] key={key_heads} value={value_heads} tokens={tokens} seqs={num_seqs}")

    # Ancla absoluta: el kernel FP32 contra el oraculo FP32 (no cambia con B2).
    out_ref32, st_ref32 = run_reference(ref, inp, make_state(inp, torch.float32, device, 7))
    out32, st32 = run_b12x(api, inp, make_state(inp, torch.float32, device, 7),
                           caps_kwargs=caps, override=None)
    check("fp32 salida vs oraculo", out32, out_ref32, 0.02)
    check("fp32 final vs oraculo", st32[inp["final"]], st_ref32[inp["final"]], 0.02)
    check("fp32 ckpt vs oraculo", st32[inp["ckpt"]], st_ref32[inp["ckpt"]], 0.02)

    # La prueba de B2: mismo estado de partida BF16 en el oraculo y en el kernel.
    out_ref16, st_ref16 = run_reference(ref, inp, make_state(inp, torch.bfloat16, device, 7))
    out16, st16 = run_b12x(api, inp, make_state(inp, torch.bfloat16, device, 7),
                           caps_kwargs=caps, override=None)
    check("bf16 salida vs oraculo-bf16", out16, out_ref16, 0.03)
    check("bf16 final vs oraculo-bf16", st16[inp["final"]], st_ref16[inp["final"]], 0.03,
          exact_dtype=torch.bfloat16)
    check("bf16 ckpt vs oraculo-bf16", st16[inp["ckpt"]], st_ref16[inp["ckpt"]], 0.03,
          exact_dtype=torch.bfloat16)

    # Coherencia entre bases: la diferencia bf16-vs-fp32 es el redondeo de la
    # base de partida, no un error de tipo; margen mayor pero pequeno.
    check("bf16 final vs bf16(fp32)", st16[inp["final"]], bf16_round(st32[inp["final"]]), 0.10)
    check("bf16 salida vs fp32 salida", out16, out32, 0.05)

    if do_chunk:
        check_chunk(api, ref, tuning, caps,
                    make_inputs(key_heads, value_heads, 64, 2, device, seed=13), tp, device)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prueba GPU del prefill b12x en BF16 (B2)")
    ap.add_argument("--key-heads", type=int, default=None,
                    help="fuerza cabezas clave locales (value = 3*key); sin esto TP=2 y TP=1")
    ap.add_argument("--no-chunk", action="store_true",
                    help="omite el intento chunk_parallel (solo la ruta sequential)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("B2 necesita GPU: corre esto dentro de la imagen en un Spark.")
    device = torch.device(args.device)
    torch.cuda.set_device(device if device.index is not None else 0)
    log(f"dispositivo {device} · {torch.cuda.get_device_name(device)}")

    root = patched_root(Path(__file__).resolve().parent)
    api, ref, tuning = load_patched(root)

    # Control de que el parche surte efecto: el Caps original rechaza BF16.
    try:
        api.Caps(device=device, max_tokens=16, max_seqs=1, max_state_slots=4,
                 key_heads=1, value_heads=3, state_dtype=torch.bfloat16)
    except Exception as exc:  # noqa: BLE001
        fail(f"Caps no acepta state_dtype=bfloat16 (el parche B2 no surtio efecto): {exc}")
        return 1
    ok("Caps B2 acepta state_dtype=bfloat16")

    ranges = ([(args.key_heads, 3 * args.key_heads, "forzado")]
              if args.key_heads else TP_RANGES)
    for key_heads, value_heads, tp in ranges:
        try:
            check_case(api, ref, tuning, key_heads, value_heads, tp, device,
                       do_chunk=not args.no_chunk)
        except Exception as exc:  # noqa: BLE001
            fail(f"{tp}: {type(exc).__name__}: {exc}")

    log("")
    if SKIPS:
        log(f"SALTOS ({len(SKIPS)}):")
        for s in SKIPS:
            log(f"  - {s}")
    if FAILURES:
        log(f"RESULTADO: {len(FAILURES)} FALLO(S)")
        for f in FAILURES:
            log(f"  - {f}")
        return 1
    log("RESULTADO: B2 PREFILL BF16 OK" + ("" if not SKIPS else " (con saltos, mirarlos)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
