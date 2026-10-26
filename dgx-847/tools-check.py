#!/usr/bin/env python3
"""DGX-847: el parser de tools de la receta de ursu (qwen3_xml + reasoning qwen3) en las dos formas en que lo usan
Hermes (chat completions con tools, en stream, a través de LiteLLM) y Claude Code (/v1/messages, si el servidor lo sirve).
  python3 tools-check.py http://HOST:8000 OUT_DIR
Escribe OUT_DIR/tools-check.json y sale 0 si todo pasa. Claude Code se prueba aparte (chain.sh), solo si /v1/messages da 200.
"""
import json, os, sys, time, urllib.request
from pathlib import Path

BASE, OUT = sys.argv[1].rstrip("/"), Path(sys.argv[2])
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Tiempo actual de una ciudad",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "Lista los ficheros de un directorio",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
                    "required": ["path"]}}},
]
ASK = [{"role": "user", "content": "¿Qué tiempo hace en Madrid y en París, en grados C? Usa la herramienta, una llamada por ciudad."}]


def post(path, body, stream=False):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=600)


def chat(messages, stream):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 4096, "stream": stream}
    if not stream:
        return json.load(post("/v1/chat/completions", body))["choices"][0]["message"]
    msg, calls = {"content": "", "reasoning": ""}, {}
    for line in post("/v1/chat/completions", body, True):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])["choices"]
        if not d:
            continue
        delta = d[0].get("delta") or {}
        msg["content"] += delta.get("content") or ""
        msg["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
        for tc in delta.get("tool_calls") or []:
            c = calls.setdefault(tc["index"], {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            c["id"] = tc.get("id") or c["id"]
            f = tc.get("function") or {}
            c["function"]["name"] += f.get("name") or ""
            c["function"]["arguments"] += f.get("arguments") or ""
    msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg


def check_calls(msg):
    errs, cities = [], []
    calls = msg.get("tool_calls") or []
    if not calls:
        errs.append("sin tool_calls")
    for c in calls:
        try:
            a = json.loads(c["function"]["arguments"])
            cities.append(a.get("city"))
        except Exception as e:
            errs.append(f"argumentos no JSON: {c['function']['arguments']!r} ({e})")
        if c["function"]["name"] != "get_weather":
            errs.append(f"tool inesperada {c['function']['name']}")
    if any(t in (msg.get("content") or "") for t in ("<tool_call>", "<function=", "<parameter=", "</think>")):
        errs.append("marcas del parser filtradas al content")
    return errs, cities


res, ok = {}, True
for stream in (False, True):
    t = time.time()
    m = chat(ASK, stream)
    errs, cities = check_calls(m)
    # segunda vuelta: resultados de las tools y respuesta final en texto
    follow = ASK + [{"role": "assistant", "content": None, "tool_calls": m.get("tool_calls") or []}] + [
        {"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"temp_c": 21 + i, "sky": "despejado"})}
        for i, c in enumerate(m.get("tool_calls") or [])]
    final = chat(follow, stream) if not errs else {}
    if not errs and (final.get("tool_calls") or not (final.get("content") or "").strip()):
        errs.append("la segunda vuelta no acaba en texto")
    if any(t in (final.get("content") or "") for t in ("<tool_call>", "<function=", "</think>")):
        errs.append("marcas filtradas al content final")
    res["stream" if stream else "no-stream"] = {"ok": not errs, "errors": errs, "cities": cities, "n_calls": len(m.get("tool_calls") or []),
                                                 "final": (final.get("content") or "")[:300], "s": round(time.time() - t, 1)}
    ok &= not errs

try:
    r = post("/v1/messages", {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "di hola"}]})
    res["anthropic_messages"] = r.status
except Exception as e:
    res["anthropic_messages"] = repr(e)[:200]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tools-check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False))
sys.exit(0 if ok else 1)
