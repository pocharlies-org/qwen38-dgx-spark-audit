#!/usr/bin/env python3
"""Comparador greedy token a token para QWEN_GDN_MIX (DGX-852).

Lanza los mismos casos contra UN endpoint y guarda las salidas (capture), o
lanza los mismos casos contra otro endpoint y las compara con lo guardado
(compare). Temperature=0 y `enable_thinking:false` siempre. No toca el cluster:
solo HTTP.

Por que capture/compare y no dos endpoints a la vez: hacen falta dos servidores
con el MISMO TP para que la comparacion sea entre kernels y no entre paralelos
(TP cambia el reparto de cabezas y el orden de los all-reduce, y eso domina la
divergencia). Con tres Sparks solo hay un 2x, asi que E2 se lanza, se captura,
se apaga, se lanza A en el mismo puesto y se compara contra el fichero.

Casos (los que pidio Dani):
  short<i>                 greedy corto, sin nada especial.
  long-chunked-prefill     prompt calibrado por encima de
                           --max-num-batched-tokens (8192) -> prefill troceado.
  cache-cold / cache-warm  el prompt largo dos veces seguidas en el MISMO
                           servidor: la segunda pisa la caché de prefijo
                           alineada. Su diferencia ENTRE SI es la prueba de que
                           el estado guardado en caché es correcto, sin
                           comparar con nadie (ver PLAN-GPU.md).
  prefix-cache-turn2       multi-vuelta: turno 1, luego turno 2 con el
                           historial; se guarda solo el turno 2.
  mixed-cN-<i>             N peticiones IDENTICAS a la vez (c8): el scheduler
                           mezcla prefill+decode; ahi A/B caen al camino triton
                           (no existe b12x_mixed con prefill triton).

Los prompts se construyen en la pasada `capture` y se guardan en el JSON: la
pasada `compare` usa EXACTAMENTE los mismos textos, si no la calibracion del
largo cambiaria entre pasadas y la comparacion no significaria nada.

Criterio (PLAN-GPU.md): el primer token generado depende solo del prefill, asi
que una divergencia en posicion 0-1 con texto raro = estado corrompido. Una
divergencia tardia y fluida = diferencia numerica entre kernels, o ruido de
batching: por eso hay que capturar E2 DOS VECES (suelo de ruido) antes de
juzgar A o B.

Uso:
  python3 -I greedy_compare.py --url http://spark2:8000 --capture e2-a.json
  python3 -I greedy_compare.py --url http://spark2:8000 --capture e2-b.json
  python3 -I greedy_compare.py --url http://spark2:8000 --compare e2-a.json \
      --noise e2-b.json --out informe-A.json
Sale 0 si ningun caso es `suspecto`; 1 si alguno lo es; 2 error de red/uso.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = "Eres un asistente tecnico preciso. Responde en castellano."


def post(url: str, path: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (URL fija)
        return json.loads(r.read().decode())


def chat(url: str, model: str, messages: list, max_tokens: int, timeout: int) -> str:
    resp = post(url, "/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout)
    return resp["choices"][0]["message"].get("content") or ""


def tokenize(url: str, model: str, text: str, timeout: int) -> list[int]:
    try:
        r = post(url, "/tokenize", {"model": model, "prompt": text}, timeout)
        return [t["id"] if isinstance(t, dict) else t for t in r.get("tokens", [])]
    except Exception:
        # Sin /tokenize (detras de LiteLLM): ids = codigos de char; la divergencia
        # de texto se ve igual, solo que en unidades de caracter.
        return [ord(c) for c in text]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def degenerate(text: str) -> bool:
    """Salida corrupta: vacia, o un fragmento repetido hasta el infinito (el
    colapso tipico de un estado recurrente pisado)."""
    if not text.strip():
        return True
    tail = text[-120:]
    for size in (16, 12, 8):
        if len(tail) >= 2 * size and tail.count(tail[-size:]) >= 3:
            return True
    return False


def build_long_prompt(url: str, model: str, target_tokens: int, timeout: int) -> str:
    unit = ("Explica con detalle como un almacen de claves distribuido mantiene "
            "la coherencia ante fallos, sin encabezados y sin concluir. ")
    prompt = unit
    step = unit
    for _ in range(400):
        try:
            n = len(tokenize(url, model, prompt, timeout))
        except Exception:
            n = len(prompt) // 4
        if n >= target_tokens:
            break
        prompt += step
        step = step * 2 if len(step) < 4000 else step
    return prompt


def case_single(url: str, model: str, prompt: str, max_tokens: int, timeout: int):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = chat(url, model, msgs, max_tokens, timeout)
    return text, tokenize(url, model, text, timeout)


def case_multiturn(url, model, p1, p2, max_tokens, timeout):
    """Turno 1 (rellena la cache) y turno 2 con el historial; se guarda el 2."""
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p1}]
    r1 = chat(url, model, m, max_tokens, timeout)
    m2 = m + [{"role": "assistant", "content": r1}, {"role": "user", "content": p2}]
    r2 = chat(url, model, m2, max_tokens, timeout)
    return r2, tokenize(url, model, r2, timeout)


def run_cases(url: str, args, prompts: dict) -> dict:
    """Ejecuta todos los casos contra `url` y devuelve {caso: {text, ids}}."""
    res: dict[str, dict] = {}

    def store(name, pair):
        text, ids = pair
        res[name] = {"text": text, "ids": ids}

    for i, p in enumerate(prompts["short"]):
        store(f"short{i}", case_single(url, args.model, p, args.max_tokens, args.timeout))

    store("long-chunked-prefill",
          case_single(url, args.model, prompts["long"], args.max_tokens, args.timeout))
    # el mismo prompt otra vez, sin tocar nada: la segunda pisa la cache de prefijo
    store("cache-warm",
          case_single(url, args.model, prompts["long"], args.max_tokens, args.timeout))
    res["cache-cold"] = dict(res["long-chunked-prefill"])

    store("prefix-cache-turn2",
          case_multiturn(url, args.model, prompts["turn1"], prompts["turn2"],
                         args.max_tokens, args.timeout))

    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = [ex.submit(case_single, url, args.model, prompts["mixed"],
                          args.max_tokens, args.timeout) for _ in range(args.conc)]
        pairs = [f.result() for f in futs]
    for i, pair in enumerate(pairs):
        store(f"mixed-c{args.conc}-{i}", pair)

    return res


def make_prompts(url: str, args) -> dict:
    return {
        "short": list(args.short_prompts),
        "long": build_long_prompt(url, args.model, args.long_tokens, args.timeout),
        "turn1": args.turn1,
        "turn2": args.turn2,
        "mixed": args.mixed_prompt,
    }


def do_capture(url: str, args) -> int:
    prompts = make_prompts(url, args)
    print(f"capture {url} -> {args.capture} (long={len(prompts['long'])} chars)")
    out = {"url": url, "model": args.model, "conc": args.conc,
           "prompts": prompts, "cases": run_cases(url, args, prompts)}
    Path(args.capture).write_text(json.dumps(out, ensure_ascii=False))
    print(f"# capturados {len(out['cases'])} casos en {args.capture}")
    return 0


def do_compare(url: str, args) -> int:
    base = json.loads(Path(args.compare).read_text())
    prompts = base["prompts"]
    args.conc = base.get("conc", args.conc)
    noise = None
    if args.noise:
        noise = json.loads(Path(args.noise).read_text())["cases"]
    print(f"compare {url} contra {args.compare}"
          + (f" (suelo de ruido {args.noise})" if noise else ""))
    now = run_cases(url, args, prompts)

    rows = []
    for name, ref in base["cases"].items():
        cur = now.get(name)
        if cur is None:
            rows.append({"case": name, "suspect": True, "why": "caso ausente"})
            continue
        fd = first_divergence(ref["ids"], cur["ids"])
        # self-check: en ESTA pasada, el resultado con cache debe ser igual al
        # frio. Es interno al servidor: no depende de E2 ni del ruido entre lanzamientos.
        self_fd = None
        if name in ("long-chunked-prefill", "cache-cold"):
            self_fd = first_divergence(now["cache-cold"]["ids"], now["cache-warm"]["ids"])
        floor = None
        if noise is not None and name in noise:
            floor = first_divergence(ref["ids"], noise[name]["ids"])
        early = fd is not None and fd < args.early_tokens
        # `floor` NO es una puerta: es cuanto se repite a si mismo E2. Si E2 ya
        # divergio de E2 en este caso (floor no None), el numero de A no dice
        # nada y solo vale la degeneracion y el frio/caliente.
        non_deterministic = floor is not None
        self_bad = self_fd is not None and self_fd < args.early_tokens
        suspect = degenerate(cur["text"]) or self_bad \
            or (early and not non_deterministic)
        rows.append({
            "case": name, "first_divergence": fd, "identical": fd is None,
            "noise_floor": floor, "cold_vs_warm": self_fd,
            "cand_degenerate": degenerate(cur["text"]), "suspect": suspect,
            "ref_head": ref["text"][:60], "cand_head": cur["text"][:60],
        })
        flag = "SUSPECTO" if suspect else ("identico" if fd is None else f"diverge@{fd}")
        extra = f" suelo@{floor}" if floor is not None else ""
        extra += f" frio/caliente@{self_fd}" if self_fd is not None else ""
        print(f"  {name}: {flag}{extra}")

    report = {"url": url, "baseline": args.compare, "noise": args.noise, "rows": rows,
              "suspects": [r["case"] for r in rows if r.get("suspect")]}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"# escrito {args.out}")
    if report["suspects"]:
        print(f"RESULTADO: {len(report['suspects'])} caso(s) SUSPECTOS: "
              + ", ".join(report["suspects"]))
        return 1
    print("RESULTADO: sin sospechas de corrupcion")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="endpoint a atacar en esta pasada")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", metavar="JSON", help="guarda las salidas aqui")
    g.add_argument("--compare", metavar="JSON", help="baseline guardado con --capture")
    ap.add_argument("--noise", metavar="JSON", help="segunda captura E2: suelo de ruido")
    ap.add_argument("--out", default=None, help="donde escribir el informe de compare")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long-tokens", type=int, default=16000,
                    help="objetivo de tokens del prompt largo (por encima de 8192)")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--early-tokens", type=int, default=8,
                    help="divergencia por debajo de esto = sospecha de corrupcion")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--turn1", default="Tengo un cluster k3s con tres nodos Spark. Que es el arbitro GPU?")
    ap.add_argument("--turn2", default="Y ahora, que pasa si un nodo se queda termico? Responde breve.")
    ap.add_argument("--mixed-prompt", default="Nombra tres algoritmos de consenso y su diferencia, en una frase cada uno.")
    ap.add_argument("--short-prompts", nargs="*", default=[
        "Explica que es la cuantificacion NVFP4 en una frase.",
        "Para que sirve un all-reduce en entrenamiento distribuido?",
        "Resume en dos lineas que hace un kernel de atencion dispersa.",
    ])
    args = ap.parse_args()
    return do_capture(args.url, args) if args.capture else do_compare(args.url, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"error de red: {exc}", file=sys.stderr)
        raise SystemExit(2)
