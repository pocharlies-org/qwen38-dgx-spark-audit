#!/usr/bin/env python3
"""DGX-852: micro-repro del IMA del MoE b12x nvfp4 en UN Spark, sin el modelo entero.

Construye el bloque MoE de UNA capa tal y como lo ve el rank 0 de TP=3 sin EP
(I_pp=768/3=256, n=2*I_pp=512, k=hidden=2560, E=512, topk=10) con los pesos NVFP4
reales de la vista e768, a traves del MISMO codigo vLLM del arranque A:
Qwen4ExpSparseMoeBlock + ModelOptMixedPrecision (hf_quant_config.json) + B12xExperts +
process_weights_after_loading. tp_size=3/tp_rank=0 se inyecta parcheando las dos
funciones que leen el grupo (layer.py:50 y config.py:1124); el grupo distributivo real
es world=1, asi que el all-reduce final es no-op y solo importa la particion de pesos.

Criterios (stdout):
  1. --backend b12x --hidden randn: se espera CRASH en M=6 (primera ejecucion del micro
     nvfp4 con enrutado real; M=8..56 son dynamic y pasaron las 9 capturas del arranque A).
  2. --backend b12x --hidden zeros (como el warmup: router sobre ceros -> topk 0..9): no
     debe crashar -> fija la diferencia warmup/captura en la distribucion de expertos.
  3. Con B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=0 en el entorno: no debe crashar ningun M ->
     el workaround del bloque-wa.sh funciona (todo dynamic, kernel ya probado con enrutado
     real y compilado independiente de m: _impl.py:9411-9433).
  4. --ref: referencia fp32 con dequant nvfp4 de los pesos. El camino b12x cuantiza
     tambien las activaciones (nvfp4 static), asi que 0.01-0.1 es normal; >0.2 delata un
     bug de verdad (p.ej. el swap gate/up del layout "w13" de la glue en --layer mtp).
  5. --backend flashinfer_cutlass guarda las salidas en /tmp/repro-*.pt para el diff
     entre backends del Job (misma semilla -> misma ruta del router).

El crash del arranque A es en el paso EAGER que precede a la captura
(cudagraph_utils.py:386), por eso esto no toca CUDA graphs: forward eager con
CUDA_LAUNCH_BLOCKING=1 (se fuerza aqui antes de importar torch).
"""
import argparse
import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import json  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/cache/huggingface/qwen38fn-mtp-ursu-nvfp4-tp3-e768")
    p.add_argument("--layer", default="0", help="'0' = capa target 0; 'mtp' = mtp.layers.0")
    p.add_argument("--backend", default="b12x", help="moe_backend: b12x|flashinfer_cutlass|auto")
    p.add_argument("--linear", default="b12x", help="linear_backend denso (igual que el arranque A)")
    p.add_argument("--sizes", default="1,2,3,4,6,8,12,16,24,30,32,40,48,56")
    p.add_argument("--hidden", default="randn", choices=["randn", "zeros"],
                   help="randn = enrutado real; zeros = como el warmup b12x (expertos 0..9)")
    p.add_argument("--ref", action="store_true", help="comparar contra dequant fp32 en el primer M")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ckpt_weights(model_path, prefix):
    idx = json.load(open(f"{model_path}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    names = sorted(n for n in wm if n.startswith(prefix + "."))
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(wm[n], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(f"{model_path}/{shard}", framework="pt") as f:
            for n in ns:
                yield n[len(prefix) + 1:], f.get_tensor(n)


def dequant_nvfp4(weight_u8, scale_u8, scale_2):
    """[O,K/2] uint8 (2 codigos e2m1 por byte, nibble bajo primero) + [O,K/16] e4m3."""
    lut = torch.tensor(FP4_LUT)
    lo = weight_u8 & 0x0F
    hi = weight_u8 >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(*weight_u8.shape[:-1], -1)
    vals = lut[codes.long()]
    s = scale_u8.view(torch.float8_e4m3fn).float()
    s = s.repeat_interleave(16, dim=-1)
    return vals * s * float(scale_2)


def reference_moe(block, model_path, layer_prefix, hidden, E, topk, K, I_pp, cache):
    """MoE fp32 con dequant nvfp4 de los pesos del checkpoint (rank 0: columnas 0..I_pp)."""
    h = hidden.float().cpu()
    logits = h @ block.gate.weight.float().cpu().t()
    probs = torch.softmax(logits, dim=-1)
    w, ids = torch.topk(probs, topk, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)  # renorm (norm_topk_prob=True)
    out = torch.zeros_like(h)
    for e in ids.unique().tolist():
        if e not in cache:
            def grab(suffix, _e=e):
                for n, t in ckpt_weights(model_path, f"{layer_prefix}.experts.{_e}"):
                    if n == suffix:
                        return t
                raise KeyError(suffix)
            gw = grab("gate_proj.weight"); gs = grab("gate_proj.weight_scale"); g2 = grab("gate_proj.weight_scale_2")
            uw = grab("up_proj.weight"); us = grab("up_proj.weight_scale"); u2 = grab("up_proj.weight_scale_2")
            dw = grab("down_proj.weight"); ds = grab("down_proj.weight_scale"); d2 = grab("down_proj.weight_scale_2")
            gate = dequant_nvfp4(gw, gs, g2)[:I_pp]
            up = dequant_nvfp4(uw, us, u2)[:I_pp]
            down = dequant_nvfp4(dw, ds, d2)[:, :I_pp]
            cache[e] = (gate, up, down)
        gate, up, down = cache[e]
        mask = ids == e
        rows = torch.where(mask.any(dim=-1))[0]
        for t in rows.tolist():
            j = int(torch.where(ids[t] == e)[0][0])
            inter = torch.nn.functional.silu(h[t] @ gate.t()) * (h[t] @ up.t())
            out[t] += w[t, j] * (inter @ down.t())
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    from vllm.config import (CompilationConfig, KernelConfig, ModelConfig,
                             ParallelConfig, VllmConfig, set_current_vllm_config)
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.layers.fused_moe import config as fm_config
    from vllm.model_executor.layers.fused_moe import layer as fm_layer

    fm_layer.get_tensor_model_parallel_world_size = lambda: 3
    fm_config.get_tensor_model_parallel_rank = lambda: 0

    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment)
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.transformers_utils.config import get_config

    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method="tcp://127.0.0.1:29517",
                                 local_rank=0, backend="nccl")
    ensure_model_parallel_initialized(1, 1)

    hf = get_config(args.model, trust_remote_code=True)
    text_cfg = getattr(hf, "text_config", hf)
    e768 = int(text_cfg.moe_intermediate_size)
    assert e768 % 3 == 0, f"la vista debe tener moe_intermediate % 3 == 0, es {e768}"
    E, topk, K = int(text_cfg.num_experts), int(text_cfg.num_experts_per_tok), int(text_cfg.hidden_size)
    I_pp = e768 // 3
    print(f"config: E={E} topk={topk} k={K} moe_intermediate={e768} -> rank0 I_pp={I_pp} n={2*I_pp}")

    model_config = ModelConfig(model=args.model, trust_remote_code=True, dtype="bfloat16")
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=1),
        kernel_config=KernelConfig(moe_backend=args.backend, linear_backend=args.linear),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE.value),
    )

    layer_prefix = (f"model.language_model.layers.{args.layer}.mlp"
                    if args.layer.isdigit() else "mtp.layers.0.mlp")
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vllm_config):
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock
        block = Qwen4ExpSparseMoeBlock(vllm_config=vllm_config, prefix=layer_prefix)
        loaded = AutoWeightsLoader(block).load_weights(ckpt_weights(args.model, layer_prefix))
        print(f"cargadas {len(loaded)} entradas bajo {layer_prefix}")
        block.eval()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"backend={args.backend} linear={args.linear} hidden={args.hidden} sizes={sizes} "
          f"cutover={os.environ.get('B12X_MICRO_DYNAMIC_CUTOVER_PAIRS', '(default 64)')} "
          f"force_a16={os.environ.get('VLLM_B12X_MOE_FP4_FORCE_A16', '0')}")
    crash = None
    outs = {}
    dequant_cache: dict = {}
    for m in sizes:
        hidden = (torch.zeros(m, K, device=dev, dtype=torch.bfloat16) if args.hidden == "zeros"
                  else torch.randn(m, K, device=dev, dtype=torch.bfloat16) * 0.1)
        try:
            with torch.inference_mode():
                out = block(hidden)
            torch.cuda.synchronize()
            logits = hidden @ block.gate.weight.t()
            ids = torch.topk(logits.float(), topk, dim=-1).indices
            n_active = len(ids.unique())
            print(f"M={m:>3}: OK (expertos activos: {n_active})")
            outs[m] = out.float().cpu()
            if args.ref and m == sizes[0]:
                ref = reference_moe(block, args.model, layer_prefix, hidden,
                                    E, topk, K, I_pp, dequant_cache)
                rel = (ref - out.float().cpu()).abs().max() / ref.abs().max().clamp(min=1e-6)
                print(f"REF M={m}: max rel error = {rel:.3e} "
                      f"(<0.1 ok; >0.2 = bug de verdad, p.ej. layout w13/w31)")
        except Exception as e:  # noqa: BLE001
            torch.cuda.synchronize()
            print(f"M={m:>3}: CRASH {type(e).__name__}: {str(e)[:400]}")
            crash = m
            break
    torch.save(outs, f"/tmp/repro-{args.backend}-{args.hidden}-L{args.layer}.pt")
    print("RESULTADO:", f"CRASH en M={crash}" if crash else "sin crash en ninguno")
    return 1 if crash else 0


if __name__ == "__main__":
    raise SystemExit(main())
