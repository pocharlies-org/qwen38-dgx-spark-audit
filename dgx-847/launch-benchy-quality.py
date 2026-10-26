#!/usr/bin/env python3
"""DGX-847 (copia de dgx-820; BASE, RES, MODEL, ARM, DEPLOY y DEADLINE por env; la calidad se corta en DEADLINE):
DGX-817 (qa): las dos patas que faltan del metodo comun contra el residente TP=3, en secuencia y nada concurrente.
  1. llama-benchy medium (pp 2048, tg 128, depth 0/65535, c1/4/8, 3 runs, --enable-prefix-caching) -> tp3-benchy/medium_tp3.csv
  2. bateria de calidad tools/qwen38_quality (mismos argumentos que res1 de DGX-743)           -> tp3-quality/
La bateria tiene ventana prohibida 03:00-09:00 Europe/Madrid (el proxy responde 503): si la corrida (tope 6 h) la cruzaria,
espera al fin de la ventana (09:00) antes de empezar. Escribe DONE (primera linea exit=N) al final.
Codigos: 0 ok; 2 residente sin /health; 3 tope de benchy; 4 benchy fallo; 5 la bateria fallo; 130 senal.
  setsid nohup bash -c 'python3 ~/bench-results/dgx-820/launch-benchy-quality.py > ~/bench-results/dgx-820/run-benchy-quality.log 2>&1' >/dev/null 2>&1 &
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

H = Path.home()
BASE = Path(os.environ["BASE"])
RES = os.environ["RES"]
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
ARM, DEPLOY = os.environ["ARM"], os.environ["DEPLOY"]
TICKET, ROL = os.environ.get("TICKET", "DGX-847"), os.environ.get("ROL", "qa")  # quien firma la nota y en qué ticket
DEADLINE = float(os.environ["DEADLINE"])  # epoch: la ventana se cierra a esa hora, la calidad no pasa de ahi
BENCHY = H / "bin" / "benchy-venv" / "bin" / "llama-benchy"
TOKENIZER = H / "benchy-tokenizer" / "qwen38-fn-hybrid"
OB, OQ = BASE / "benchy", BASE / "quality"
SRC = H / "bench-results" / "dgx-818" / "src"  # el src con --min-max-tokens que produjo dgx-817/tp3-quality
BENCHY_CAP = 75 * 60
QUALITY_CAP = 6 * 3600
MAD = ZoneInfo("Europe/Madrid")
KUBE = dict(os.environ, KUBECONFIG=str(H / ".kube/config"))
stop = threading.Event()
cur = {"p": None}
code = {"v": 0, "reason": "ok"}
T0 = datetime.now(timezone.utc)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    print(f"[{now()}] {m}", flush=True)


def metrics():
    try:
        txt = urllib.request.urlopen(RES + "/metrics", timeout=8).read().decode()
        g = lambda n: next((float(m.group(1)) for m in re.finditer(r"^vllm:%s\{[^}]*\}\s+([0-9.eE+-]+)" % n, txt, re.M)), None)
        return g("num_requests_running"), g("num_requests_waiting")
    except Exception:
        return None, None


def sampler(path, ev):
    while not ev.is_set():
        r, w = metrics()
        with open(path, "a") as f:
            f.write(json.dumps({"t": now(), "running": r, "waiting": w}) + "\n")
        ev.wait(5)


def kube(args, out=None):
    try:
        o = subprocess.run(["kubectl"] + args, capture_output=True, text=True, timeout=60, env=KUBE).stdout
        if out:
            Path(out).write_text(o)
        return o
    except Exception as e:
        return repr(e)


def snapshot(d, tag):
    p = kube(["get", "pods", "-n", "llm", "-o", "jsonpath={range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount}/{.status.containerStatuses[0].state.running.startedAt} {end}"])
    a = kube(["get", "application", "-n", "argocd", "ai", "-o", "jsonpath={.spec.source.targetRevision} {.status.sync.status} {.status.health.status}"])
    with open(d / "pods.jsonl", "a") as f:
        f.write(json.dumps({"t": now(), "tag": tag, "ai": a, "pods": [x for x in p.split() if ("qwen38" in x or "ursu" in x)]}) + "\n")


def run(cmd, log_path, cap, env=None):
    with open(log_path, "a") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env or os.environ, start_new_session=True)
        cur["p"] = p
        try:
            return p.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            return "cap"


def on_signal(sig, _f):
    stop.set()
    if cur["p"]:
        try:
            os.killpg(cur["p"].pid, signal.SIGTERM)
        except Exception:
            pass
    finish(130, "senal")


def finish(c, reason):
    stop.set()
    code["v"], code["reason"] = c, reason
    end = datetime.now(timezone.utc)
    (BASE / "DONE-benchy-quality").write_text(f"exit={c}\nreason={reason}\nstarted={T0:%Y-%m-%dT%H:%M:%SZ}\nended={end:%Y-%m-%dT%H:%M:%SZ}\n")
    attach(c, reason)
    sys.exit(c)


def attach(c, reason):
    try:
        nota = BASE / f"nota-qa-{ARM}-benchy-calidad.md"
        nota.write_text(
            f"Rol: {ROL} · Fecha: {now()} · Sesión: 5d35c318-9261-4879-b52f-24fa3c4db837 · Estado: BORRADOR\n\n"
            f"# {TICKET}: benchy medium y batería de calidad contra la receta de ursu {ARM}\n\n"
            f"Fin del script: exit={c} ({reason}).\n\n"
            f"- benchy: {OB}/medium_{ARM}.csv (run.log al lado; DONE-benchy con su exit)\n"
            f"- calidad: {OQ} (stages.jsonl, proxy.jsonl, items-*.jsonl; lectura en el siguiente despacho de qa)\n")
        subprocess.run(["python3", str(H / ".local/libexec/x86-host-runtime/jira_rest.py"), "attach", "--rol", ROL, TICKET, str(nota)],
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        log(f"attach fallo: {e!r}")


def madrid_start_ok(t, hours):
    """True si [t, t+hours] no cruza 03:00-09:00 Madrid; si no, devuelve el instante de las 09:00 que cierra la ventana."""
    end = t + timedelta(hours=hours)
    tm = t.astimezone(MAD)
    for d in range(0, 3):
        w0 = (tm + timedelta(days=d)).replace(hour=3, minute=0, second=0, microsecond=0)
        w1 = w0.replace(hour=9)
        if t < w1 and end > w0:
            return False, w1
    return True, None


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)

BASE.mkdir(exist_ok=True)
OB.mkdir(exist_ok=True)
OQ.mkdir(exist_ok=True)
try:
    ok = urllib.request.urlopen(RES + "/health", timeout=10).status == 200
except Exception:
    ok = False
if not ok:
    finish(2, "el residente no da /health")

# ---- 1. benchy
if os.environ.get("SKIP_BENCHY"):
    log("benchy: ya hecho (DONE-benchy)")
    stop_b = threading.Event()
else:
    log("benchy: inicio")
    snapshot(OB, "inicio")
    kube(["get", "pod", "-n", "llm", "-l", f"app={DEPLOY}", "-o", "json"], OB / "pod-snapshot.json")
    kube(["exec", "-n", "llm", f"deploy/{DEPLOY}", "--", "printenv"], OB / "env-snapshot.txt")
    try:
        (OB / "models.json").write_text(urllib.request.urlopen(RES + "/v1/models", timeout=10).read().decode())
    except Exception as e:
        (OB / "models.json").write_text(repr(e))
    stop_b = threading.Event()
    threading.Thread(target=sampler, args=(OB / "load-samples.jsonl", stop_b), daemon=True).start()
    (OB / "run.log").write_text("")
    rc = run([str(BENCHY), "--base-url", RES + "/v1", "--model", MODEL, "--served-model-name", MODEL, "--tokenizer", str(TOKENIZER),
              "--pp", "2048", "--tg", "128", "--depth", "0", "65535", "--concurrency", "1", "4", "8", "--runs", "3",
              "--enable-prefix-caching", "--format", "csv", "--save-result", str(OB / f"medium_{ARM}.csv")], OB / "run.log", BENCHY_CAP)
    snapshot(OB, "fin")
    (BASE / "DONE-benchy").write_text(f"exit={rc if rc != 'cap' else 3}\nended={now()}\n")
    if rc == "cap":
        finish(3, "tope de benchy")
    if rc != 0:
        finish(4, f"benchy rc={rc}")
    log("benchy: ok")

# ---- 2. espera a la ventana si hace falta y bateria
stop_b.set()
t = datetime.now(timezone.utc)
QUALITY_CAP = int(min(QUALITY_CAP, DEADLINE - time.time()))
if QUALITY_CAP < 1800:
    finish(6, f"sin tiempo para la calidad antes del cierre ({QUALITY_CAP} s)")
log(f"calidad: tope {QUALITY_CAP} s (cierre de la ventana)")
# con BLACKOUT (endpoints de prueba) no hay franja que esperar
ok, until = (True, None) if os.environ.get("BLACKOUT") else madrid_start_ok(t, QUALITY_CAP / 3600)
if not ok:
    log(f"la bateria cruzaria 03:00-09:00 Madrid: espera hasta {until.astimezone(timezone.utc):%Y-%m-%dT%H:%MZ} ({until:%H:%M} Madrid)")
    while datetime.now(timezone.utc) < until and not stop.is_set():
        time.sleep(30)
(BASE / "quality-waited-until").write_text(until.isoformat() if until else "no espera")
try:
    ok = urllib.request.urlopen(RES + "/health", timeout=10).status == 200
except Exception:
    ok = False
if not ok:
    finish(2, "el residente no da /health antes de la calidad")
log("calidad: inicio")
snapshot(OQ, "inicio")
kube(["get", "pod", "-n", "llm", "-l", f"app={DEPLOY}", "-o", "json"], OQ / "pod-snapshot.json")
threading.Thread(target=sampler, args=(OQ / "load-samples.jsonl", stop), daemon=True).start()
env = dict(os.environ, HF_HOME=str(H / "hf-qwen38-quality"), HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
rc = run([sys.executable, str(SRC / "tools/qwen38_quality/run_battery.py"), "--arm", ARM, "--base-url", RES, "--out", str(OQ),
          "--venv-eval", str(H / "venvs/qwen38-quality-eval"), "--venv-bfcl", str(H / "venvs/qwen38-quality-bfcl"),
          "--humaneval-image", "sha256:19657b88d97a16ce4b0601fd884655a4b140eb15633de96f6560b077fc15de82", "--priority", "100", "--min-max-tokens", "16384", "--blackout", os.environ.get("BLACKOUT", "03:00-09:00"),
          "--suites", "tools12,gsm8k,minerva_math,ifeval,bfcl,humaneval,niah131072"], OQ / "launcher.log", QUALITY_CAP, env)
snapshot(OQ, "fin")
if rc != 0:
    finish(5, f"bateria rc={rc}")
finish(0, "ok")
