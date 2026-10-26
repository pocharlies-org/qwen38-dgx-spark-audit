#!/usr/bin/env python3
"""La sonda de refusal (FASE B), en CPU y sin vLLM.

Lo que se comprueba, en el orden en que puede romperse:
  1. probe() escribe el coseno en la columna de SU capa, ignora capas no
     sondeadas y el drafter MTP, y NO toca el tensor del forward.
  2. step_stats() reparte bien los tokens de un lote entre peticiones de
     longitud distinta (el fallo tipico: un offset corrido le da a una peticion
     los tokens de su vecina, y sin mirar AMBAS no se ve).
  3. accumulate() separa prompt de generacion, se queda con el ultimo token del
     ULTIMO tramo de prefill y corta la generacion en GEN_WINDOW.
  4. Sin calibracion: score presente, flag None, cal None. Nunca un veredicto
     inventado.
  5. Con calibracion: flag segun umbral y op; una calibracion invalida NO da
     flag; y se relee por mtime sin reiniciar.
  6. Forma inesperada: la sonda se desactiva (fail-safe), no revienta.
"""
import json
import math
import os
import pathlib
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "payload"))
import refusal_probe as P  # noqa: E402

H = 2560
fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")


class Req:
    pass


def cos(x, d):
    x = x.double()
    return float((x @ d.double()) / x.norm())


def main():
    torch.manual_seed(0)
    d = torch.randn(H)
    d = d / d.norm()
    P.reset()
    P.init_with_direction(d, layers=(16, 24))

    # 1 -- probe
    x = torch.randn(5, H, dtype=torch.bfloat16)
    x_before = x.clone()
    P.probe(x, 16, "model.layers.16")
    P.probe(x * 2, 24, "model.layers.24")        # coseno invariante a escala
    P.probe(torch.randn(5, H), 20, "model.layers.20")   # capa no sondeada
    buf = P._P.buf
    check("probe no modifica el tensor del forward", torch.equal(x, x_before))
    check("capa 16 -> columna 0, coseno exacto",
          all(abs(float(buf[i, 0]) - cos(x[i].float(), d)) < 1e-4 for i in range(5)))
    check("capa 24 -> columna 1, invariante a escala",
          all(abs(float(buf[i, 1]) - cos(x[i].float(), d)) < 1e-4 for i in range(5)))
    # tronco hiper-conectado: 4 streams de H, se suma y luego coseno
    hs = torch.randn(5, 4 * H, dtype=torch.bfloat16)
    hs_before = hs.clone()
    P.probe(hs, 16, "model.layers.16", 4)
    summed = hs.float().reshape(5, 4, H).sum(1)
    check("hc=4: coseno de la suma de streams, tensor intacto",
          torch.equal(hs, hs_before)
          and all(abs(float(buf[i, 0]) - cos(summed[i], d)) < 1e-4 for i in range(5)))
    P.probe(x, 16, "model.layers.16")
    snap = buf[:5].clone()
    P.probe(torch.randn(5, H), 16, "mtp.layers.16")
    check("el drafter MTP no pisa la capa 16 del target", torch.equal(buf[:5], snap))

    # 2 -- step_stats: dos peticiones, 3 y 2 tokens
    vals = torch.tensor([[0.1, 0.3], [0.5, 0.1], [0.2, 0.2],     # req 0
                         [-0.4, 0.0], [0.6, 0.8]])               # req 1
    buf[:5].copy_(vals)
    qsl = torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32)       # con padding
    out = P.step_stats(5, 2, qsl, [True, False])
    rp = P.materialize(out)
    s = vals.mean(1).tolist()
    r0, r1 = rp["rows"]
    check("req 0: max/suma/n de SUS tres tokens",
          abs(r0[0] - max(s[:3])) < 1e-6 and abs(r0[1] - sum(s[:3])) < 1e-6 and r0[2] == 3)
    check("req 1: max/suma/n de SUS dos tokens",
          abs(r1[0] - max(s[3:])) < 1e-6 and abs(r1[1] - sum(s[3:])) < 1e-6 and r1[2] == 2)
    check("ultimo token por capa de cada peticion",
          [round(v, 6) for v in r0[3:]] == [0.2, 0.2]
          and [round(v, 6) for v in r1[3:]] == [0.6, 0.8], f"{r0[3:]} {r1[3:]}")
    check("prefill por peticion", rp["prefill"] == [True, False])

    # 3 -- accumulate: prefill en dos tramos + decode
    req = Req()
    P.accumulate(req, {"rows": [[0.9, 1.0, 4, 0.1, 0.1]], "prefill": [True]}, 0)
    P.accumulate(req, {"rows": [[0.3, 0.6, 2, 0.2, 0.4]], "prefill": [True]}, 0)
    for _ in range(40):   # 40 pasos x 4 tokens = 160 > GEN_WINDOW
        P.accumulate(req, {"rows": [[0.05, 0.08, 4, 0.0, 0.0]], "prefill": [False]}, 0)
    acc = req._refusal_probe_acc
    check("prompt: max de todos los tramos", abs(acc["p_max"] - 0.9) < 1e-9)
    check("prompt: ultimo token = el del ULTIMO tramo", acc["p_last"] == [0.2, 0.4])
    check("generacion cortada en GEN_WINDOW", acc["g_n"] == P.GEN_WINDOW, f"g_n={acc['g_n']}")
    P.accumulate(req, None, 0)
    P.accumulate(req, {"rows": [], "prefill": []}, 3)   # indice fuera: no revienta

    # 4 -- sin calibracion
    os.environ.pop("VLLM_REFUSAL_PROBE_CAL", None)
    v = P.finalize(req)
    check("sin cal: score = prompt_last_mean", v and abs(v["score"] - 0.3) < 1e-6, f"{v}")
    check("sin cal: flag None y cal None", v["flag"] is None and v["cal"] is None)
    check("finalize limpia el acumulador", not hasattr(req, "_refusal_probe_acc"))
    check("finalize sin tokens -> None", P.finalize(Req()) is None)
    check("gen_mean correcto", abs(v["gen_mean"] - 0.02) < 1e-6, f"{v['gen_mean']}")
    json.dumps(v)   # tiene que viajar por msgspec/JSON tal cual

    # 5 -- calibracion con relectura por mtime
    with tempfile.TemporaryDirectory() as tmp:
        cal = pathlib.Path(tmp) / "cal.json"
        cal.write_text(json.dumps({"id": "t1", "feature": "prompt_last_mean",
                                   "threshold": 0.25, "op": ">="}))
        os.environ["VLLM_REFUSAL_PROBE_CAL"] = str(cal)
        P._cal_cache.update(path=None, mtime=None, checked=0.0, cal=None)

        def verdict(last):
            r = Req()
            P.accumulate(r, {"rows": [[0.0, 0.0, 1] + last], "prefill": [True]}, 0)
            return P.finalize(r)

        a, b = verdict([0.3, 0.3]), verdict([0.1, 0.1])
        check("cal: por encima del umbral -> flag True", a["flag"] is True and a["cal"] == "t1")
        check("cal: por debajo -> flag False", b["flag"] is False)

        time.sleep(0.01)
        cal.write_text(json.dumps({"id": "t2", "feature": "prompt_last_mean",
                                   "threshold": 0.05, "op": ">="}))
        os.utime(cal, (time.time() + 5, time.time() + 5))
        P._cal_cache["checked"] = 0.0
        c = verdict([0.1, 0.1])
        check("cal releida por mtime sin reiniciar", c["flag"] is True and c["cal"] == "t2", f"{c}")

        cal.write_text(json.dumps({"id": "t3", "feature": "inventada", "threshold": 1}))
        os.utime(cal, (time.time() + 10, time.time() + 10))
        P._cal_cache["checked"] = 0.0
        e = verdict([0.9, 0.9])
        check("cal invalida -> sin flag (nunca un veredicto inventado)",
              e["flag"] is None and e["cal"] is None, f"{e}")
        os.environ.pop("VLLM_REFUSAL_PROBE_CAL", None)

    # 6 -- forma inesperada: se desactiva, no revienta
    P.probe(torch.randn(3, H // 2), 16, "model.layers.16")
    check("forma inesperada desactiva la sonda", not P.is_active())
    check("desactivada: step_stats -> None", P.step_stats(5, 2, qsl, [True, True]) is None)
    P.probe(torch.randn(3, H), 16, "model.layers.16")   # y sigue sin reventar

    # 7 -- apagada por entorno: probe no hace nada y no inicializa
    P.reset()
    os.environ.pop("VLLM_REFUSAL_PROBE", None)
    P.probe(torch.randn(3, H), 16, "model.layers.16")
    check("sin VLLM_REFUSAL_PROBE no se activa", not P.is_active())
    check("capas por entorno", P.parse_layers("12, 30,12") == (12, 30)
          and P.parse_layers("") == P.DEFAULT_LAYERS)
    check("nan/inf no viajan como numero", P._r(math.inf) is None and P._r(float("nan")) is None)

    print(f"\n{fails} fallos")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
