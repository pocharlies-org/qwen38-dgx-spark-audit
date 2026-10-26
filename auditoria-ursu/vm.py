import json, subprocess, sys, datetime
from urllib.parse import urlencode
P = "/api/v1/namespaces/monitoring/services/http:vmsingle-vm-victoria-metrics-k8s-stack:8428/proxy/api/v1/query_range"
def q(expr, start, end, step):
    u = P + "?" + urlencode({"query": expr, "start": start, "end": end, "step": step})
    out = subprocess.run(["kubectl", "get", "--raw", u], capture_output=True, text=True, timeout=60)
    if out.returncode: sys.exit(out.stderr)
    return json.loads(out.stdout)["data"]["result"]
W = sys.argv[1] if len(sys.argv) > 1 else "1h"
start, end = "2026-10-24T23:00:00Z", "2026-10-26T22:00:00Z"
by = 'service'
S = '{service=~"qwen38-flash-next-head|qwen38-flash-next-tp1"}'
M = {
 "dec_stream": f'sum by ({by}) (increase(vllm:request_generation_tokens_sum{S}[{W}])) / sum by ({by}) (increase(vllm:request_decode_time_seconds_sum{S}[{W}]))',
 "agregado":   f'sum by ({by}) (increase(vllm:generation_tokens_total{S}[{W}])) / {W[:-1]}/3600' if W.endswith('h') else None,
 "ttft_s":     f'sum by ({by}) (increase(vllm:time_to_first_token_seconds_sum{S}[{W}])) / sum by ({by}) (increase(vllm:time_to_first_token_seconds_count{S}[{W}]))',
 "cola_s":     f'sum by ({by}) (increase(vllm:request_queue_time_seconds_sum{S}[{W}])) / sum by ({by}) (increase(vllm:request_queue_time_seconds_count{S}[{W}]))',
 "prefill_tps":f'sum by ({by}) (increase(vllm:request_prefill_kv_computed_tokens_sum{S}[{W}])) / sum by ({by}) (increase(vllm:request_prefill_time_seconds_sum{S}[{W}]))',
 "peticiones": f'sum by ({by}) (increase(vllm:request_decode_time_seconds_count{S}[{W}]))',
 "running":    f'avg by ({by}) (avg_over_time(vllm:num_requests_running{S}[{W}]))',
 "waiting":    f'avg by ({by}) (avg_over_time(vllm:num_requests_waiting{S}[{W}]))',
 "acept_mtp":  f'sum by ({by}) (increase(vllm:spec_decode_num_accepted_tokens_total{S}[{W}])) / sum by ({by}) (increase(vllm:spec_decode_num_drafts_total{S}[{W}]))',
}
rows = {}
for name, expr in M.items():
    if not expr: continue
    for s in q(expr, start, end, W):
        svc = s["metric"].get(by, "?").replace("qwen38-flash-next-", "")
        for t, v in s["values"]:
            rows.setdefault((svc, int(t)), {})[name] = float(v)
cols = [c for c in M if M[c]]
print("svc  hora_utc     " + " ".join(f"{c:>11}" for c in cols))
for (svc, t) in sorted(rows):
    r = rows[(svc, t)]
    ts = datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%m-%d %H:%M")
    print(f"{svc:4} {ts}  " + " ".join(f"{r.get(c, float('nan')):11.1f}" for c in cols))
