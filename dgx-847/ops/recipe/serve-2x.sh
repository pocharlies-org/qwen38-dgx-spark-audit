#!/bin/bash
S=/opt/b12x-seed/preparation; D="$XDG_CACHE_HOME/b12x/compile/preparation"; if [ -d "$S" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$D" && for f in "$S"/*.json; do [ -e "$D/$(basename "$f")" ] || cp "$f" "$D/"; done; fi
A=/opt/b12x-seed/torch_compile_cache; if [ -d "$A" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p "$XDG_CACHE_HOME/vllm/torch_compile_cache" && cp -rn "$A/." "$XDG_CACHE_HOME/vllm/torch_compile_cache/" || true; fi
M=/cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd; [ -s $M/model-00035-of-00036.safetensors ] && [ -s $M/model.safetensors.index.json ] || HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download as d; d('local-inference-lab/Qwen3.8-Flash-Next-NVFP4', revision='7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd', allow_patterns=['model-00035-of-00036.safetensors', 'model.safetensors.index.json'])" || exit 1
exec vllm serve ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --revision 03f4a0570496bbe741189f4a2a7d611679e85406 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name qwen38-flash-next qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-parallel-prefills 4 \
  --prefill-policy decode-aware \
  --decode-refill-target auto \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format b12x \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  --scheduling-policy priority \
  --disable-access-log-for-endpoints /metrics,/health \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
  --gpu-memory-utilization 0.75 \
  --nnodes 2 --node-rank $NODE_RANK --master-addr 10.0.0.2 --master-port 25000 $HEADLESS ${EXTRA_ARGS:-}
