---
pretty_name: Qwen3.8-Flash-Next on 3x DGX Spark — speed audit
license: mit
tags:
  - benchmark
  - dgx-spark
  - vllm
  - qwen
  - inference
language:
  - en
  - es
---

# Qwen3.8-Flash-Next on 3× NVIDIA DGX Spark — speed audit & recipe comparison

Raw data, scripts and logs from a speed audit of our resident LLM (Qwen3.8-Flash-Next,
served by a vLLM fork — "ursu" images, TP=2 across two DGX Sparks + TP=1 on a third,
k3s, RoCE ring) run on 2026-10-26: did the resident get faster or slower over the last
two weeks, and how does **our recipe** compare to the stock [ursu 2x v2.1.0 recipe](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2)?

Same image (`13460354`), same two Sparks. The difference is the launch recipe.

## Our recipe vs stock ursu

| | Ursu (FP32 state) | Ours (BF16 state + triton prefill) | Δ |
|---|---|---|---|
| Prefix-cache block | 2864 tokens | 1504 | ~half |
| 64K cached + 2K new — first answer, 1 req | 2.23 s | 1.67 s | **−25 %** |
| Same, 8 concurrent reqs | 10.63 s | 9.00 s | **−15 %** |
| Generation @ 64K context, 8 reqs | 58 tok/s | 74 | **+27 %** |
| Generation, 1 req | 85 | 90 | +6 % |
| Cold prefill 2K, 1 req | 2946 tok/s | 2481 | **−16 %** |

DGX-847, run E2, 2026-10-13, same machines both sides.

**Why ours wins for agentic traffic:** in production 86 % of input tokens come from the
prefix cache (avg request 55K input tokens, only ~7.5K computed fresh; median 1.8K).
Cached conversations recompute from the last complete block — with 1504-token blocks we
recalculate about half of what ursu's 2864-token blocks do. The −16 % cold-prefill loss
is only paid on the first prompt of each conversation.

Other deltas: fixed 29.5 GB KV per rank (ursu: 80 % of memory), 10 concurrent seqs
(ursu: 16 — long prefills don't collapse: 16.1K → 5.7K tok/s from 8→16 seqs), Spanish
MTP draft vocab, priority scheduler, refusal probe (λ=0).

## HC in BF16 (hyperconnection down-projection)

−3–4 % cold-prefill time on requests with >~1.3K new tokens, decode within noise,
bit-identical output. Worth it; not worth building a custom 2x image for.

## Caveats

- The −25 % / −15 % table is from synthetic prompts (2026-10-13); the production cache
  pattern was measured 2026-10-26 but the A/B was not re-run on real traffic.
- No isolated A/B for the refusal probe or the Spanish draft vocab.
- There is no permanent benchmark system here; these are one-off audits, kept raw.

## Contents

- `dgx-847/` — the main audit: specs, test scripts (`prueba-*.sh`, `measure.py`),
  `mtp/definitivo/` final runs (ursu vs A vs B2, prefill/decode JSON+logs), quality
  checks, HTML reports.
- `dgx-932/` — B2 (b12x prefill BF16) TP=2 measurements: 68/162/230/341 tok/s at
  1/4/8/16 reqs vs 73/166/218/341 for A.
- `auditoria-ursu/` — before/after GitOps manifests and the live-flags diff of our TP=2
  process vs the ursu v2.1.0 recipe (`diff/nuestro-flags.txt` vs `diff/ursu-flags.txt`).

Internal hostnames/IPs are replaced with placeholders (`spark1..3`, `tailnet-*`,
`internal.example`). No secrets: everything was env-var references or comments.

Dataset mirror: https://huggingface.co/datasets/pocharlies/qwen38-dgx-spark-audit

License: MIT (scripts) — data is provided as-is, no warranty.
