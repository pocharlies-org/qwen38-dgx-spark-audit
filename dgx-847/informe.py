#!/usr/bin/env python3
"""DGX-847: informe HTML autocontenido (un fichero, sin dependencias) con todo lo medido: decode, benchy, calidad y las
pruebas A-G/C. Lee los ficheros de resultados; lo que aún no exista sale como «pendiente». El texto de conclusiones se
toma de conclusiones.html (si existe).
  python3 -I informe.py            (escribe informe-dgx-847.html junto al script)
  python3 -I informe.py completo   (informe-qwen38-spark.html: lo mismo más la sección de DGX-852, dgx852.html de dgx852.py)"""
import csv, html, json, sys, re
from pathlib import Path

D = Path(__file__).resolve().parent
B17 = D.parent / "dgx-817"
P = D / "pruebas"
BRAZOS = {"tp3": ("Nuestro TP=3 (3 Sparks)", B17 / "tp3-01f9b4d-valla", B17 / "tp3-benchy/medium_tp3.csv"),
          "2x": ("Su TP=2 (dgx1+dgx2)", D / "2x/decode", D / "2x/benchy/medium_2x.csv"),
          "1x": ("Su 1 Spark (dgx3)", D / "1x/decode", D / "1x/benchy/medium_1x.csv")}
e = html.escape


def lee(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def benchy(p):
    try:
        return {r["test_name"]: r for r in csv.DictReader(open(p))}
    except Exception:
        return {}


def f(x, nd=1, suf=""):
    return "—" if x in (None, "") else f"{float(x):,.{nd}f}{suf}".replace(",", " ")


def tabla(cab, filas, cls=""):
    h = "".join(f"<th>{c}</th>" for c in cab)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>" for fila in filas)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def barras(titulo, datos, unidad, mejor_alto=True):
    """datos: [(etiqueta, valor)], barras horizontales en CSS."""
    datos = [(k, v) for k, v in datos if v is not None]
    if not datos:
        return ""
    m = max(v for _, v in datos) or 1
    best = (max if mejor_alto else min)(v for _, v in datos)
    filas = "".join(f'<div class="b"><span class="bl">{e(k)}</span><span class="bt"><span class="bf{" best" if v == best else ""}" '
                    f'style="width:{100 * v / m:.1f}%"></span></span><span class="bv">{f(v)} {unidad}</span></div>' for k, v in datos)
    return f'<div class="chart"><h4>{e(titulo)}</h4>{filas}</div>'


secc = []
# ---- decode v4
dec = {k: lee(v[1] / "base-63dd55f-agg.json") for k, v in BRAZOS.items()}
g = lee(P / "g-ple-disco/decode-agg.json")
if g:
    dec["g"] = g
g2 = lee(P / "g2-ple-disco-gmu07/decode-agg.json")
if g2:
    dec["g2"] = g2
g1 = lee(P / "1x/g-ple-gpu/decode-agg.json")
if g1:
    dec["g1"] = g1
nom = {**{k: v[0] for k, v in BRAZOS.items()}, "g": "Su TP=2 con la PLE en disco (prueba G)", "g1": "Su 1 Spark con la PLE en GPU (G al revés)",
       "g2": "Su TP=2 con la PLE en disco y gpu-mem 0,7 (G2; c1/c8 ×2)"}
cs = ["1", "4", "8", "10"]
filas = [[e(nom[k])] + [f((d or {}).get("summary", {}).get(c, {}).get("agg_tok_s_median")) for c in cs] for k, d in dec.items()]
mtp = (dec["2x"] or {}).get("summary", {})
secc.append(("Generación (decode v4, tok/s totales)",
             "<p>Peticiones cortas, 256 tokens de salida, 3 repeticiones de 120 s por concurrencia (mediana).</p>"
             + tabla(["Receta", "c1", "c4", "c8", "c10"], filas)
             + '<div class="charts">' + "".join(barras(f"c{c}", [(nom[k], (d or {}).get("summary", {}).get(c, {}).get("agg_tok_s_median")) for k, d in dec.items()], "tok/s") for c in ("1", "8")) + "</div>"
             + "<p>Aceptación del MTP en su TP=2 (k=4, por posición, c1): "
             + " · ".join(f(v, 2) for v in mtp.get("1", {}).get("mtp_accept_by_pos_median", {}).values()) + "</p>"))
# ---- 64K a c8/c10 (decode v4, contexto largo en frío)
filas = []
for k, v in BRAZOS.items():
    r = [e(v[0])]
    for c in ("8", "10"):
        x = lee(v[1] / f"base-63dd55f-64k-c{c}.json")
        r.append(f((x or {}).get("summary", {}).get(c, {}).get("ttft_p50_median"), 1, " s"))
    filas.append(r)
secc.append(("Contexto de 64K en frío (primera respuesta, p50)", tabla(["Receta", "c8", "c10"], filas)))
# ---- benchy
bz = {k: benchy(v[2]) for k, v in BRAZOS.items()}
filas_def = [("tg128 (c1)", "t_s_mean", "Generación c1 (tok/s)"), ("tg128 (c4)", "t_s_mean", "Generación c4 (tok/s)"),
             ("tg128 (c8)", "t_s_mean", "Generación c8 (tok/s)"), ("pp2048 (c1)", "t_s_mean", "Prefill 2K c1 (tok/s)"),
             ("tg128 @ d65535 (c1)", "t_s_mean", "64K: generación c1 (tok/s)"), ("tg128 @ d65535 (c8)", "t_s_mean", "64K: generación c8 (tok/s)"),
             ("pp2048 @ d65535 (c1)", "e2e_ttft_mean", "64K con caché: 1.ª respuesta c1 (s)"),
             ("pp2048 @ d65535 (c8)", "e2e_ttft_mean", "64K con caché: 1.ª respuesta c8 (s)")]
filas = [[e(t)] + [f(float(bz[k][n][col]) / (1000 if col == "e2e_ttft_mean" else 1), 1 if col != "e2e_ttft_mean" else 2) if n in bz[k] and bz[k][n][col] else "—"
                   for k in BRAZOS] for n, col, t in filas_def]
secc.append(("llama-benchy (medium, con caché de prefijo)", tabla(["Medida"] + [e(v[0]) for v in BRAZOS.values()], filas)))
# ---- calidad
cal = lee(D / "calidad-vs-tp3.json") or {}
filas = []
for s, d in cal.items():
    k = d["k"]
    cel = lambda n: f"{k[n][0]}/{k[n][1]} ({100 * k[n][0] / k[n][1]:.1f} %)" if n in k else "—"
    dl = lambda n: (f"{d[n + '_vs_tp3']['delta'] * 100:+.1f} pp" + (" *" if d[n + "_vs_tp3"]["excluye0"] else "")) if n + "_vs_tp3" in d else "—"
    filas.append([e(s), cel("tp3"), cel("2x"), cel("1x"), dl("2x"), dl("1x")])
secc.append(("Calidad (7 suites, pareada ítem a ítem contra nuestro TP=3)",
             tabla(["Suite", "TP=3", "TP=2", "1 Spark", "TP=2 − TP=3", "1 Spark − TP=3"], filas)
             + "<p>* = el intervalo de confianza del 95 % excluye el 0. La única diferencia real es <code>minerva_math_em</code>, que puntúa el "
               "formato exacto de la respuesta; la nota matemática (<code>minerva_math</code>, math-verify) empata.</p>"))
# ---- pruebas
def tabla_a(a):
    return tabla(["Paso", "Tokens", "Reutilizados", "Recalculados", "1.ª respuesta"],
                 [[e(x["paso"]), f(x["prompt_tokens"], 0), f(x["reutilizados_metricas"], 0), f(x["recalculados"], 0), f(x["ttft_s"], 2, " s")] for x in a["pasos"]])


def celda(r, n, c):
    return f(float(r[n][c]) / (1000 if c == "e2e_ttft_mean" else 1), 2) if n in r and r[n][c] else "—"


EXTRA_2X = (("e2-bf16-gdn-triton", "E2 · Estado SSM en BF16 con el GDN estándar (triton) en vez de b12x: sin recompilar, bloque de 1 504 tokens"),
            ("g2-ple-disco-gmu07", "G2 · PLE en disco con <code>--gpu-memory-utilization 0.7</code> (memoria de sobra para la caché de páginas)"))


def pruebas(Pd, a_json, base, receta, bloque, kv_base, g_tag, g_tit, g_cols, c_txt, extra=()):
    """Bloque de pruebas A-B-E-F-G-C (y las extra) de una receta; lo que falte sale como pendiente."""
    pr = []
    a = lee(a_json)
    pr.append(f"<h3>A · Reutilización de la caché sobre 64K ({receta} tal cual)</h3>"
              + (tabla_a(a) + f"<p>La caché guarda el estado en bloques de {bloque} tokens (lo fija el tamaño de página del estado SSM en float32): "
                 f"con 2 048 tokens nuevos sobre 64K reaprovecha {f(a['pasos'][1]['reutilizados_metricas'], 0)} y recalcula {f(a['pasos'][1]['recalculados'], 0)}.</p>"
                 if a else "<p class='pend'>pendiente</p>"))
    for tag, tit in (("b-mnbt4096", "B · <code>--max-num-batched-tokens 4096</code> (el nuestro; el suyo es 8192)"),
                     ("e-ssm-bf16", "E · Estado SSM en BF16 (nuestra palanca más fuerte)"),
                     ("f-nuestros-3", "F · Nuestras palancas del planificador: 4096 por paso + 2048 por petición (sin BF16, imposible con b12x)")) + tuple(extra):
        r = benchy(Pd / tag / "medium.csv")
        ac = lee(Pd / tag / "a-cache-ctx65535.json")
        if not r:
            log = Pd / tag / "run.log"
            txt = log.read_text() if log.exists() else ""
            err = [l for l in txt.splitlines() if "ValueError" in l or "no arranca" in l or "exigen" in l]
            pr.append(f"<h3>{tit}</h3><p class='pend'>{e(' · '.join(err)[:400]) if err else 'pendiente'}</p>")
            continue
        filas = []
        for n, col, t in (filas_def if tag in dict(EXTRA_2X) else filas_def[4:]):
            v0 = float(base[n][col]) if n in base and base[n][col] else None
            v1 = float(r[n][col]) if n in r and r[n][col] else None
            filas.append([e(t), celda(base, n, col), celda(r, n, col), f(100 * (v1 - v0) / v0 if v0 and v1 else None, 0, " %")])
        extra = ""
        if ac:
            x = ac["pasos"][1]
            extra = f"<p>Prueba A con esta configuración: reutiliza {f(x['reutilizados_metricas'], 0)}, recalcula {f(x['recalculados'], 0)}, 1.ª respuesta {f(x['ttft_s'], 2)} s.</p>"
        pr.append(f"<h3>{tit}</h3>" + tabla(["Medida", "Su receta", "Con el cambio", "Δ (en tiempos, − es mejor)"], filas) + extra)
    gb = benchy(Pd / g_tag / "medium.csv")
    arr = Pd / g_tag / "arranque.log"
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", arr.read_text()) if arr.exists() else []
    gl = Pd / g_tag / "run.log"
    pr.append(f"<h3>{g_tit}</h3>"
              + (("<p>Decode en la tabla de generación de arriba. Caché KV: " + (e(kv[0]) + f" tokens (su receta: {kv_base})" if kv else "—") + ".</p>"
                  + tabla(["Medida"] + g_cols, [[e(t), celda(base, n, c), celda(gb, n, c)] for n, c, t in filas_def]))
                 if gb else f"<p class='pend'>{e(gl.read_text().splitlines()[0]) if gl.exists() and 'no arranca' in gl.read_text() else 'pendiente'}</p>"))
    pr.append("<h3>C · Perfil (dónde se va el tiempo de GPU con 64K en caché + 2 048 nuevos, c1 y c8)</h3>"
              + (f"<pre>{e(c_txt.read_text().replace(str(D) + '/', ''))}</pre>" if c_txt.exists() else "<p class='pend'>pendiente</p>"))
    return "".join(pr)


secc.append(("Pruebas de contexto largo y PLE · su TP=2",
             "<p><b>Pregunta:</b> ¿por qué nuestro TP=3 responde antes con 64K de contexto ya en caché, y puede la PLE ir a disco sin perder velocidad?</p>"
             + pruebas(P, P / "a-cache-2x-ctx65535.json", bz["2x"], "su TP=2", "2 864", "3 419 333", "g-ple-disco",
                       "G · La tabla PLE en disco (su <code>VLLM_PLE_MMAP</code>, el mismo mecanismo que nuestro DGX-773)",
                       ["PLE en GPU (su receta)", "PLE en disco"], P / "c-perfil/resumen.txt", EXTRA_2X)))
P1 = P / "1x"
secc.append(("Las mismas pruebas · su 1 Spark (dgx3, en paralelo)",
             pruebas(P1, P1 / "a-cache-ctx65535.json", bz["1x"], "su 1 Spark", "3 024", "993 754", "g-ple-gpu",
                     "G al revés · La tabla PLE en GPU (su 1 Spark la lleva en disco): lo que cuesta el disco",
                     ["PLE en disco (su receta)", "PLE en GPU"], P1 / "c-perfil/resumen.txt")))
# ---- tools
tl = []
for k in ("2x", "1x"):
    t = lee(D / k / "tools-check.json") or {}
    cc = (D / k / "claude-code.json")
    ok = "—"
    if cc.exists():
        ok = "sí" if '"is_error":false' in cc.read_text().replace(" ", "") else "no"
    tl.append([e(BRAZOS[k][0]), "sí" if t.get("no-stream", {}).get("ok") else "no", "sí" if t.get("stream", {}).get("ok") else "no",
               e(str(t.get("anthropic_messages", "—"))), ok])
secc.append(("Herramientas", tabla(["Receta", "Tools sin stream", "Tools en stream", "/v1/messages", "Claude Code"], tl)))

concl = (D / "conclusiones.html").read_text() if (D / "conclusiones.html").exists() else "<p class='pend'>Conclusiones al terminar las pruebas.</p>"
COMPLETO = sys.argv[1:] == ["completo"]
if COMPLETO:
    secc.insert(0, ("DGX-852 · MTP más largo y su checkpoint NVFP4 sobre nuestra pila (en curso)", (D / "dgx852.html").read_text()))
cuerpo = "".join(f"<section><h2>{t}</h2>{c}</section>" for t, c in secc)
out = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.8-Flash-Next en 3 DGX Spark: recetas de ursuciprian frente a nuestro TP=3</title><style>
:root{{--bg:#0f1115;--card:#171a21;--tx:#e6e8ee;--mu:#9aa3b2;--ac:#76b900;--ln:#2a2f3a}}
body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:auto;padding:28px 18px 60px}} h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;border-bottom:2px solid var(--ac);padding-bottom:6px;margin-top:0}}
h3{{font-size:16px;margin:22px 0 8px}} h4{{margin:0 0 6px;font-size:13px;color:var(--mu)}} .sub{{color:var(--mu);margin-bottom:22px}}
section{{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:18px 20px;margin:18px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}} th,td{{padding:6px 9px;border-bottom:1px solid var(--ln);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--mu);font-weight:600}} code{{background:#232838;padding:1px 5px;border-radius:4px;font-size:13px}}
pre{{background:#0b0d11;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px}} .pend{{color:#e0a43a}} a{{color:var(--ac)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin:12px 0}}
.b{{display:grid;grid-template-columns:150px 1fr 90px;gap:8px;align-items:center;font-size:12px;margin:3px 0}} .bl{{color:var(--mu)}}
.bt{{background:#232838;border-radius:4px;height:14px}} .bf{{display:block;height:14px;border-radius:4px;background:#4a5568}} .bf.best{{background:var(--ac)}} .bv{{text-align:right}}
</style></head><body><main>
<h1>Qwen3.8-Flash-Next en 3 DGX Spark</h1>
<div class="sub">Recetas de ursuciprian (fork de vLLM con kernels b12x, MTP k=4, NVFP4) frente a nuestro qwen38 en TP=3 · DGX-847 · 12-13/10/2026<br>
<a href="https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2">receta TP=2</a> · <a href="https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark">receta 1 Spark</a> ·
<a href="https://www.reddit.com/r/DGX_Spark/s/vPijahNDQS">post</a>{"<br>Incluye DGX-852 (13/10/2026): lo que traemos de su receta a nuestra pila y hasta dónde hemos llegado." if COMPLETO else ""}</div>
<section><h2>Conclusiones</h2>{concl}</section>{cuerpo}</main></body></html>"""
dest = D / ("informe-qwen38-spark.html" if COMPLETO else "informe-dgx-847.html")
dest.write_text(out)
print(dest, len(out))
