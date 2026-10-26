#!/usr/bin/env python3
"""DGX-847 prueba A: cuánto reutiliza la caché de prefijo sobre un contexto de ~64K y cuánto tarda la primera respuesta.
  python3 prueba-a-cache.py http://HOST:8000 OUT.json
Tres peticiones en serie, con el servidor SIN otro tráfico (los contadores de /metrics son del servidor entero):
  1. P = contexto de ~64K + pregunta            (llena la caché)
  2. P + 2048 tokens nuevos + pregunta          (lo que mide benchy pp2048 @ d65535)
  3. la 2 exacta otra vez                       (techo: todo salvo el último bloque)
Reutilizados = delta de vllm:prefix_cache_hits_total; si el servidor da usage.prompt_tokens_details.cached_tokens, también.
"""
import json, random, re, sys, time, urllib.request

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 65535  # tokens del contexto (benchy d65535 lo deja en 65535)
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
rnd = random.Random(847)
SIL = ["ka", "lo", "mi", "ru", "te", "sa", "no", "vi", "de", "pa", "zo", "fe", "ti", "gu", "ba", "re"]


def texto(n_palabras):
    return " ".join("".join(rnd.choice(SIL) for _ in range(rnd.randint(2, 4))) for _ in range(n_palabras))


def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=900)


def ntok(msgs):
    return json.load(post("/tokenize", {"model": MODEL, "messages": msgs}))["count"]


def metricas():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    g = lambda n: sum(float(m.group(1)) for m in re.finditer(r"^vllm:%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)" % n, t, re.M))
    return g("prefix_cache_hits_total"), g("prefix_cache_queries_total")


def chat(msgs):
    body = {"model": MODEL, "messages": msgs, "max_tokens": 8, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0, ttft, usage = time.time(), None, None
    for line in post("/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if ttft is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None:
            ttft = time.time() - t0
        usage = d.get("usage") or usage
    return ttft, usage


# contexto de ~64K tokens: se ajusta con /tokenize
ctx = texto(30000)
while ntok([{"role": "user", "content": ctx}]) < CTX:
    ctx += " " + texto(3000)
# recorta por palabras hasta quedar justo en CTX: el sitio donde acaba el contexto respecto al bloque de la caché es lo que se mide
pal = ctx.split(" ")
lo, hi = 1, len(pal)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if ntok([{"role": "user", "content": " ".join(pal[:mid])}]) <= CTX: lo = mid
    else: hi = mid - 1
ctx = " ".join(pal[:lo])
nuevo = texto(900)
while ntok([{"role": "user", "content": nuevo}]) < 2048:
    nuevo += " " + texto(100)
Q = "\n\nResponde solo: OK."
pasos = [("1 contexto 64K", ctx + Q), ("2 contexto + 2048 nuevos", ctx + "\n" + nuevo + Q), ("3 repetición exacta de 2", ctx + "\n" + nuevo + Q)]
res = {"base": BASE, "model": MODEL, "pasos": []}
for nombre, contenido in pasos:
    msgs = [{"role": "user", "content": contenido}]
    n = ntok(msgs)
    h0, q0 = metricas()
    ttft, usage = chat(msgs)
    h1, q1 = metricas()
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    r = {"paso": nombre, "prompt_tokens": n, "reutilizados_metricas": int(h1 - h0), "consultados_metricas": int(q1 - q0),
         "cached_tokens_usage": cached, "recalculados": n - int(h1 - h0), "ttft_s": round(ttft or -1, 3)}
    res["pasos"].append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
