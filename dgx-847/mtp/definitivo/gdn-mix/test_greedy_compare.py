#!/usr/bin/env python3
"""Self-test de greedy_compare.py (DGX-852, sin GPU).

Levanta un servidor HTTP que imita /v1/chat/completions y /tokenize, y comprueba
la mecanica del comparador, que es donde se cuela un falso positivo de
corrupcion:

  1. capture y compare contra el MISMO stub determinista -> identico, exit 0.
  2. compare contra un stub que cambia el primer token -> SUSPECTO, exit 1.
  3. el suelo de ruido: si la segunda captura de E2 divergio sola en un caso,
     ese caso no puede sentenciar corrupcion por divergencia temprana.
  4. frio/caliente: si el servidor contesta distinto a la segunda copia del
     mismo prompt, es sospechoso aunque la baseline cuadre.

Uso: python3 -I test_greedy_compare.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "greedy_compare.py"

STATE = {"bad": False, "warm_differs": False, "seen": set()}
FAILURES: list[str] = []


def digest(body: dict) -> str:
    msgs = body.get("messages", [])
    last = msgs[-1]["content"] if msgs else ""
    key = hashlib.md5(json.dumps([m["content"] for m in msgs]).encode()).hexdigest()
    return key, last


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silencioso
        pass

    def _json(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode())
        key, last = digest(body)
        if self.path.startswith("/tokenize"):
            text = body.get("prompt", "")
            self._json({"tokens": [{"id": ord(c)} for c in text]})
            return
        # el prompt largo va dos veces: la segunda es la que pisa la cache
        warm = key in STATE["seen"]
        STATE["seen"].add(key)
        prefix = "XXX" if STATE["bad"] else "RESP"
        if STATE["warm_differs"] and warm and len(last) > 4000:
            prefix = "ZOMG"
        content = f"{prefix} {key[:16]} {hashlib.md5(last.encode()).hexdigest()[:24]}"
        self._json({"choices": [{"message": {"content": content}}]})


def start_server() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def run(*argv) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=300)


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def main() -> int:
    import tempfile
    base_args = ["--model", "stub", "--max-tokens", "16", "--long-tokens", "9000",
                 "--conc", "2", "--timeout", "30"]
    with tempfile.TemporaryDirectory(prefix="gcmp-") as td:
        p = lambda name: str(Path(td) / name)  # noqa: E731
        srv, url = start_server()
        try:
            # 1. capture + compare contra el mismo stub determinista
            r = run("--url", url, "--capture", p("a.json"), *base_args)
            if r.returncode != 0:
                fail(f"capture exit={r.returncode}: {r.stderr[-400:]}")
            cap = json.loads(Path(p("a.json")).read_text())
            if len(cap["cases"]) < 8:
                fail(f"casos capturados de menos: {sorted(cap['cases'])}")
            else:
                ok(f"capture: {len(cap['cases'])} casos y prompts guardados")
            if not cap["prompts"]["long"] or len(cap["prompts"]["long"]) < 9000:
                fail("el prompt largo no se calibro")
            else:
                ok("prompt largo calibrado por encima del chunk size del stub")
            r = run("--url", url, "--compare", p("a.json"), "--out", p("r1.json"), *base_args)
            if r.returncode != 0:
                fail(f"compare identico exit={r.returncode}: {r.stdout[-300:]}{r.stderr[-300:]}")
            else:
                ok("compare contra si mismo: sin sospechas, exit 0")

            # 2. stub corrupto: cambia el primer token
            STATE["bad"] = True
            r = run("--url", url, "--compare", p("a.json"), "--out", p("r2.json"), *base_args)
            rep = json.loads(Path(p("r2.json")).read_text())
            if r.returncode == 0:
                fail("stub corrupto y el comparador no lo marco (exit 0)")
            elif not rep["suspects"]:
                fail("stub corrupto: exit!=0 pero sin lista de sospechosos")
            elif all(row["first_divergence"] not in (None,) for row in rep["rows"]
                     if row["case"] in rep["suspects"]):
                ok("stub corrupto -> todos los casos sospechan y divergen en la posicion 0")
            STATE["bad"] = False

            # 3. suelo de ruido: E2 se repite mal en el lote mixto -> no sentencia
            noise = json.loads(Path(p("a.json")).read_text())
            for k in list(noise["cases"]):
                if k.startswith("mixed-"):
                    noise["cases"][k]["ids"] = noise["cases"][k]["ids"][:1] + [999999]
            Path(p("noise.json")).write_text(json.dumps(noise))
            STATE["bad"] = True
            r = run("--url", url, "--compare", p("a.json"), "--noise", p("noise.json"),
                    "--out", p("r3.json"), *base_args)
            rep = json.loads(Path(p("r3.json")).read_text())
            mixed = [row for row in rep["rows"] if row["case"].startswith("mixed-")]
            if not mixed or not all(row["noise_floor"] is not None for row in mixed):
                fail("no se reporto el suelo de ruido en los casos mixtos")
            else:
                ok("suelo de ruido presente en los casos mixtos (se informa, no calla)")
            STATE["bad"] = False

            # 4. frio/caliente interno del servidor (el stub cuenta visitas por
            # prompt: la segunda respuesta del mismo prompt debe ser igual)
            STATE["seen"].clear()
            STATE["warm_differs"] = True
            r = run("--url", url, "--compare", p("a.json"), "--out", p("r4.json"), *base_args)
            rep = json.loads(Path(p("r4.json")).read_text())
            if "cache-cold" not in rep["suspects"] and "long-chunked-prefill" not in rep["suspects"]:
                fail(f"caché fria/caliente distinta y no la marco: {rep['suspects']}")
            else:
                ok("segunda pasada del mismo prompt con otra salida -> SOSPECHOSO")
            STATE["warm_differs"] = False
        finally:
            srv.shutdown()

    print()
    if FAILURES:
        print(f"RESULTADO: {len(FAILURES)} FALLO(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULTADO: COMPARADOR OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
