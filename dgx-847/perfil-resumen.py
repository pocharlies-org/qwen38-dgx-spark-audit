#!/usr/bin/env python3
"""Suma el tiempo de GPU por kernel de cada traza torch (*.pt.trace.json.gz) y lo agrupa por familia.
  python3 perfil-resumen.py TRAZA...   (imprime una tabla por traza)"""
import gzip, json, re, sys
from collections import Counter
# el orden importa: los kernels b12x se llaman todos kernel_cutlass_kernel_b12x<familia>..., así que «cutlass» no clasifica nada
FAM = [("allreduce/nccl", r"nccl|allreduce|all_reduce|roce|custom_ar"), ("atención", r"attention|qsa|dsa|paged|flash|fmha|attn"),
       ("gdn/mamba", r"gdn|mamba|ssm|fused_recurrent|causal_conv|conv1d|gated_delta"),
       ("muestreo/mtp", r"sampl|topk|top_k|softmax|argmax|rejection|mtp|spec"), ("moe nvfp4", r"moe|fused_experts"),
       ("ple", r"ple_emb|_ple|pleemb|embedding|index_select"), ("gemm densa", r"gemm|cublas|nvjet|xmma|matmul"),
       ("norm/hiperconexión", r"norm|hyperconnection")]
for p in sys.argv[1:]:
    ev = json.load(gzip.open(p) if p.endswith(".gz") else open(p))
    ev = ev.get("traceEvents", ev)
    k = Counter()
    for e in ev:
        if e.get("cat") == "kernel" and "dur" in e:
            k[e["name"]] += e["dur"]
    tot = sum(k.values())
    if not tot:
        continue  # trazas de CPU (async_llm): sin kernels
    f = Counter()
    for n, d in k.items():
        f[next((fam for fam, rx in FAM if re.search(rx, n, re.I)), "otros")] += d
    print(f"== {p}  GPU total {tot/1e6:.2f} s")
    for fam, d in f.most_common():
        print(f"  {fam:16s} {d/1e6:8.3f} s  {100*d/tot:5.1f}%")
    for n, d in k.most_common(12):
        print(f"    {100*d/tot:5.1f}%  {n[:110]}")
