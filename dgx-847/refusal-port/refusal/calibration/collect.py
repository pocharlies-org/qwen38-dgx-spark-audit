"""Recoge (fuente, respuesta, refusal_probe) del residente para calibrar la sonda.

Los prompts salen SOLO de los dos datasets publicos estandar de la literatura de
direccion de refusal (mlabonne/harmful_behaviors y harmless_alpaca, split test),
leidos de disco. Directo al head, lambda 0, temperatura 0.

La etiqueta es lo que el modelo HIZO, no el dataset de origen: se guarda el
principio de la respuesta para etiquetarla despues (label.py).
"""
import asyncio, json, random
import httpx
import pyarrow.parquet as pq

URL = "http://10.43.157.138:8000/v1/chat/completions"
random.seed(24)
a = pq.read_table("harmful_behaviors.parquet").column("text").to_pylist()
b = random.sample(pq.read_table("harmless_alpaca.parquet").column("text").to_pylist(), len(a))
items = [("a", i, p) for i, p in enumerate(a)] + [("b", i, p) for i, p in enumerate(b)]
random.shuffle(items)


async def ask(client, sem, src, idx, prompt):
    body = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem:
        r = await client.post(URL, json=body, timeout=300)
    ch = r.json()["choices"][0]
    return {"src": src, "idx": idx, "head": (ch["message"].get("content") or "")[:300],
            "probe": ch.get("refusal_probe")}


async def main():
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(verify=False) as c:
        rows = await asyncio.gather(*[ask(c, sem, *it) for it in items])
    with open("rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "filas;", sum(r["probe"] is not None for r in rows), "con sonda")

asyncio.run(main())
