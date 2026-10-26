#!/usr/bin/env python3
"""Una línea con tok/s y aceptación del MTP por concurrencia de un decode-*.json de qwen38_decode_sweep.py."""
import json
import sys

s = json.load(open(sys.argv[1]))["summary"]
print(" · ".join(f"c{c} {v['agg_tok_s_median']:.0f}" for c, v in s.items()),
      "· acept c1", {k: round(x, 2) for k, x in (s.get("1", {}).get("mtp_accept_by_pos_median") or {}).items()})
