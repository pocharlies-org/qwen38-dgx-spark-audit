#!/usr/bin/env bash
# DGX-897: efecto observable tras 30 min de trafico con el 2:1 y el tope 16/8 (desde el despliegue de las 21:52Z).
export KUBECONFIG=~/.kube/config
sleep 1800
kubectl exec -n databases postgres-shared-2 -c postgres -- psql -d litellm -At -F ' | ' -c "
with t as (select model_id m, coalesce(session_id, metadata->>'session_id','-') sid, min(\"startTime\") t0, count(*) n, sum(prompt_tokens) pt
  from \"LiteLLM_SpendLogs\" where \"startTime\" > '2026-10-14 21:53:00+00' and model_id like 'qwen38-flash-next%' group by 1,2)
select m, count(*) sesiones, sum(n) peticiones, sum(pt) tokens from t group by 1 order by 1;" -c "
select coalesce(metadata->>'_session_routing_reason', '-') razon, count(*) from \"LiteLLM_SpendLogs\"
  where \"startTime\" > '2026-10-14 21:53:00+00' and (model_group like 'qwen38%' or model_group like 'alibaba%') group by 1 order by 2 desc limit 6;"
for h in spark2 spark3; do curl -s -m 10 http://$h:8000/metrics | awk -v h=$h '/^vllm:num_requests_running/{r+=$2} /^vllm:num_requests_waiting/{w+=$2} END{print h, "running="r, "waiting="w}'; done
