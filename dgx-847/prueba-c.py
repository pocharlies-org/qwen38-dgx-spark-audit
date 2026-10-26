#!/usr/bin/env python3
"""DGX-847 prueba C: perfil torch de la 2× en el caso que pierde frente a nuestro TP=3: contexto de 64K ya en caché + 2048
tokens nuevos, a c1 y a c8 (benchy pp2048 @ d65535). Una ventana de /start_profile../stop_profile por concurrencia.
  python3 prueba-c.py http://HOST:8000 OUT.json
"""
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
rnd = random.Random(848)
SIL = ["ka", "lo", "mi", "ru", "te", "sa", "no", "vi", "de", "pa", "zo", "fe", "ti", "gu", "ba", "re"]
texto = lambda n: " ".join("".join(rnd.choice(SIL) for _ in range(rnd.randint(2, 4))) for _ in range(n))


def post(path, body=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=1800)


def ntok(s):
    return json.load(post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": s}]}))["count"]


def chat(s):
    body = {"model": MODEL, "messages": [{"role": "user", "content": s}], "max_tokens": 32, "stream": True, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0, ttft = time.time(), None
    for line in post("/v1/chat/completions", body):
        if ttft is None and line.startswith(b"data: {") and b'"content"' in line:
            ttft = time.time() - t0
    return round(ttft or -1, 3), round(time.time() - t0, 3)


ctx = texto(30000)
while ntok(ctx) < 65535:
    ctx += " " + texto(3000)
nuevos = []
for _ in range(9):
    n = texto(900)
    while ntok(n) < 2048:
        n += " " + texto(100)
    nuevos.append(n)
Q = "\n\nResponde solo: OK."
res = {"base": BASE, "calentar": chat(ctx + Q)}  # deja el contexto en la caché, fuera del perfil
for c, lote in ((1, nuevos[:1]), (8, nuevos[1:])):
    post("/start_profile"); t0 = time.time()
    with ThreadPoolExecutor(c) as ex:
        r = list(ex.map(lambda n: chat(ctx + "\n" + n + Q), lote))
    post("/stop_profile")
    res[f"c{c}"] = {"ttft_s": [x[0] for x in r], "total_s": [x[1] for x in r], "ventana_s": round(time.time() - t0, 1)}
    print(json.dumps({f"c{c}": res[f"c{c}"]}), flush=True)
    time.sleep(30)  # que el volcado de la traza acabe antes de la siguiente ventana
json.dump(res, open(OUT, "w"), indent=1)
