#!/usr/bin/env python3
"""DGX-847 (copia de dgx-820/measure-tp3.py; RES, MODEL y OUT por env):
DGX-781: BASE del residente actual (pin 63dd55f, PLE en NVMe, TP=2) con la regla v4 de DGX-781, contra http://tailnet-node:8000.

Con trafico real (sin valla): cada punto guarda su carga ajena (running_mean, foreign_requests, contaminado) y aqui ademas
vllm:num_requests_running/waiting cada 5 s (load-base.jsonl). Pasos, en secuencia y nada concurrente contra el residente:
  1. agregado: qwen38_decode_sweep.py --concurrency 1,4,8,10 --reps 3 (args de rev5b) -> base-63dd55f-agg.json
  2. 64k c8:   --concurrency 8  --long-workers 8  --long-tokens 64000 -> base-63dd55f-64k-c8.json
  3. 64k c10:  --concurrency 10 --long-workers 10 --long-tokens 64000 -> base-63dd55f-64k-c10.json
--thermal-query incluye nvidia-dgx, gx10-ec3d y dgx3 (etiqueta `node` de kube_pod_info, leida de la metrica).
Tope duro: CAP_S (por defecto 110 min). Al acabar escribe DONE-base (primera linea exit=N) y sale con ese codigo
(0 ok, 3 tope, 4 algun paso fallo, 2 el residente no da /health, 130 senal).
  setsid nohup bash -c 'python3 ~/bench-results/dgx-781/measure-base.py > ~/bench-results/dgx-781/run-base.log 2>&1' >/dev/null 2>&1 &
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

RES = os.environ["RES"]
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
OUT = Path(os.environ["OUT"])
TOOLS = Path.home() / "bench-results" / "dgx-768" / "tools"  # copia de tools/ de k8s-ai main (la sonda no cambia entre ab24d56 y main e5b62b9)
THERMAL = "http://10.43.33.43:8428/api/v1/query"
THERMAL_Q = 'max(node_thermal_zone_temp * on(namespace, pod) group_left(node) kube_pod_info{node=~"nvidia-dgx|gx10-ec3d|dgx3"})'
CAP_S = int(os.environ.get("CAP_S", 110 * 60))
T0 = time.monotonic()
STARTED = datetime.now(timezone.utc)
state = {"code": 0, "reason": "ok", "steps": []}
stop = threading.Event()
cur = {"p": None}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    try:
        txt = urllib.request.urlopen(RES + "/metrics", timeout=8).read().decode()
        g = lambda n: next((float(m.group(1)) for m in re.finditer(r"^vllm:%s\{[^}]*\}\s+([0-9.eE+-]+)" % n, txt, re.M)), None)
        return g("num_requests_running"), g("num_requests_waiting")
    except Exception:
        return None, None


def jl(name, row):
    with open(OUT / name, "a") as f:
        f.write(json.dumps(row) + "\n")


def pods(tag):
    try:
        env = dict(os.environ, KUBECONFIG=str(Path.home() / ".kube/config"))
        o = subprocess.run(["kubectl", "get", "pods", "-n", "llm", "-o", "jsonpath={range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount}/{.status.containerStatuses[0].state.running.startedAt} {end}"],
                           capture_output=True, text=True, timeout=30, env=env).stdout
        a = subprocess.run(["kubectl", "get", "application", "-n", "argocd", "ai", "-o", "jsonpath={.spec.source.targetRevision} {.status.sync.status} {.status.health.status}"],
                           capture_output=True, text=True, timeout=30, env=env).stdout
        jl("pods-base.jsonl", {"t": now(), "tag": tag, "ai": a, "pods": [p for p in o.split() if "qwen38" in p]})
    except Exception as e:
        jl("pods-base.jsonl", {"t": now(), "tag": tag, "error": repr(e)})


def sampler():
    while not stop.wait(5):
        r, w = load()
        jl("load-base.jsonl", {"t": now(), "running": r, "waiting": w})


def finish(code=None, reason=None):
    if code is not None and state["code"] == 0:
        state["code"], state["reason"] = code, reason
    stop.set()
    p = cur["p"]
    if p and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
    pods("fin")
    (OUT / "DONE-base").write_text(
        f"exit={state['code']}\nreason={state['reason']}\nstarted={STARTED:%Y-%m-%dT%H:%M:%SZ}\nended={now()}\nelapsed_s={int(time.monotonic() - T0)}\ncap_s={CAP_S}\nsteps={json.dumps(state['steps'])}\n")
    sys.exit(state["code"])


def step(name, cmd, log):
    r, w = load()
    jl("load-base.jsonl", {"t": now(), "step": name, "running": r, "waiting": w, "before": True})
    left = CAP_S - (time.monotonic() - T0)
    if left <= 30:
        state["steps"].append({"step": name, "status": "no-iniciado-tope"})
        return None
    with open(OUT / log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        cur["p"] = p
        try:
            rc = p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            state["steps"].append({"step": name, "status": "cortado-tope", "foreign_running_before": r})
            return None
    cur["p"] = None
    state["steps"].append({"step": name, "status": "ok" if rc == 0 else f"rc={rc}", "foreign_running_before": r})
    return rc == 0


def sweep(conc, out, extra=()):
    return [sys.executable, str(TOOLS / "qwen38_decode_sweep.py"), "--base-url", RES, "--model", MODEL, "--concurrency", conc, "--warmup", "30",
            "--duration", "120", "--reps", "3", "--max-tokens", "256", "--prompts", str(TOOLS / "fixtures/qwen38_ab/prompts.jsonl"),
            "--timeout", "600", "--sample-interval", "5.0", "--long-workers", "0", "--long-tokens", "32000", "--max-model-len", "262144",
            "--seed", "496368", "--thermal-url", THERMAL, "--thermal-query", THERMAL_Q, "--out", str(out), *extra]


def with_long(cmd, workers):
    cmd = list(cmd)
    cmd[cmd.index("--long-workers") + 1] = str(workers)
    cmd[cmd.index("--long-tokens") + 1] = "64000"
    return cmd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DONE-base").unlink(missing_ok=True)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda n, _f: finish(130, f"signal {n}"))
    try:
        ok = urllib.request.urlopen(RES + "/health", timeout=10).status == 200
    except Exception:
        ok = False
    if not ok:
        finish(2, "el residente no da /health 200")
    pods("inicio")
    threading.Thread(target=sampler, daemon=True).start()
    results = [
        step("agregado", sweep("1,4,8,10", OUT / "base-63dd55f-agg.json"), "base-63dd55f-agg.log"),
        step("64k-c8", with_long(sweep("8", OUT / "base-63dd55f-64k-c8.json"), 8), "base-63dd55f-64k-c8.log"),
        step("64k-c10", with_long(sweep("10", OUT / "base-63dd55f-64k-c10.json"), 10), "base-63dd55f-64k-c10.log"),
    ]
    if any(r is None for r in results):
        finish(3, "cortado por el tope")
    bad = any(r is False for r in results)
    finish(4 if bad else 0, "algun paso fallo" if bad else "ok")


if __name__ == "__main__":
    main()
