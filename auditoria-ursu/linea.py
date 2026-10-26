import json, subprocess, sys, datetime
from urllib.parse import urlencode
P = "/api/v1/namespaces/monitoring/services/http:vmsingle-vm-victoria-metrics-k8s-stack:8428/proxy/api/v1/query_range"
def q(expr, step):
    u = P + "?" + urlencode({"query": expr, "start": "2026-10-25T12:00:00Z", "end": "2026-10-26T22:00:00Z", "step": step})
    r = subprocess.run(["kubectl", "get", "--raw", u], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout)["data"]["result"]
st = "30m"
series = {
 "tp1_vivo": 'count(vllm:num_requests_running{service="qwen38-flash-next-tp1"})',
 "h_run": 'avg_over_time(vllm:num_requests_running{service="qwen38-flash-next-head"}[30m])',
 "h_wait": 'avg_over_time(vllm:num_requests_waiting{service="qwen38-flash-next-head"}[30m])',
 "t1_run": 'avg_over_time(vllm:num_requests_running{service="qwen38-flash-next-tp1"}[30m])',
 "h_dec": 'sum(increase(vllm:request_generation_tokens_sum{service="qwen38-flash-next-head"}[30m]))/sum(increase(vllm:request_decode_time_seconds_sum{service="qwen38-flash-next-head"}[30m]))',
 "h_agr": 'sum(increase(vllm:generation_tokens_total{service="qwen38-flash-next-head"}[30m]))/1800',
 "h_ttft": 'sum(increase(vllm:time_to_first_token_seconds_sum{service="qwen38-flash-next-head"}[30m]))/sum(increase(vllm:time_to_first_token_seconds_count{service="qwen38-flash-next-head"}[30m]))',
}
rows = {}
for k, e in series.items():
    for s in q(e, st):
        for t, v in s["values"]:
            rows.setdefault(int(t), {})[k] = float(v)
print("hora_utc     " + " ".join(f"{k:>8}" for k in series))
for t in sorted(rows):
    r = rows[t]
    print(datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%m-%d %H:%M") + "  " + " ".join(f"{r.get(k, float('nan')):8.1f}" for k in series))
