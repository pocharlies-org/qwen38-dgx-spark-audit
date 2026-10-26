#!/usr/bin/env python3
"""DGX-852: la sección «MTP y su checkpoint NVFP4 sobre nuestra pila» del informe completo (dgx852.html). Lee los resultados
de mtp/ y de mtp/nvfp4/; lo que aún no exista sale como «en curso». La usa `informe.py completo`.
  python3 -I dgx852.py"""
import csv, html, json
from pathlib import Path

D = Path(__file__).resolve().parent
M = D / "mtp"
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


def tabla(cab, filas):
    h = "".join(f"<th>{c}</th>" for c in cab)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>" for fila in filas)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


BRAZOS = [("k=1 (lo que hay en producción)", M / "k1-control"), ("k=2", M / "k2"), ("k=3", M / "k3"), ("k=4", M / "k4"),
          ("NVFP4 de ursu + b12x en GEMM, k=2 (A1)", M / "nvfp4/a/k2")]
COLS = [("tg128 (c1)", "t_s_mean", "Gen. c1"), ("tg128 (c4)", "t_s_mean", "Gen. c4"), ("tg128 (c8)", "t_s_mean", "Gen. c8"),
        ("pp2048 (c1)", "t_s_mean", "Prefill 2K c1"), ("tg128 @ d65535 (c1)", "t_s_mean", "64K gen. c1"),
        ("tg128 @ d65535 (c8)", "t_s_mean", "64K gen. c8"), ("pp2048 @ d65535 (c1)", "e2e_ttft_mean", "64K 1.ª resp. c1"),
        ("pp2048 @ d65535 (c8)", "e2e_ttft_mean", "64K 1.ª resp. c8")]


def celda(r, n, c):
    if n not in r or not r[n][c]:
        return "en curso"
    v = float(r[n][c])
    return f(v / 1000, 2, " s") if c == "e2e_ttft_mean" else f(v)


filas_b = [[e(t)] + [celda(benchy(p / "medium.csv"), n, c) for n, c, _ in COLS] for t, p in BRAZOS]
filas_d = []
for t, p in BRAZOS:
    s = (lee(p / "decode-agg.json") or {}).get("summary", {})
    filas_d.append([e(t)] + [f(s.get(c, {}).get("agg_tok_s_median")) if c in s else "en curso" for c in ("1", "4", "8", "10")])
a1c = lee(M / "nvfp4/a/a-cache-ctx65535.json")
res = lee(M / "k1-control/a-cache-ctx65535.json")

out = f"""
<p><b>De dónde sale.</b> Tras DGX-847 quedó claro que la receta de ursuciprian gana en generación con un usuario (MTP k=4 barato
gracias a sus kernels b12x y a un vocabulario recortado para el borrador) y que la nuestra gana con contexto largo (estado SSM en BF16 →
bloque de caché de 832 tokens frente a 2 864). DGX-852 intenta juntar las dos: traer a nuestra pila TP=3 su checkpoint NVFP4, los
kernels b12x (solo donde no obligan a FP32) y un MTP más largo.</p>

<h3>1 · MTP más largo en nuestra pila tal cual (13-10, misma tarde, sin tráfico real)</h3>
<p>llama-benchy medium (tok/s; la 1.ª respuesta en segundos):</p>
{tabla(["Configuración"] + [c for _, _, c in COLS], filas_b)}
<p>decode v4 (tok/s totales, peticiones cortas, mediana de 3 × 120 s):</p>
{tabla(["Configuración", "c1", "c4", "c8", "c10"], filas_d)}
<p>k=2 es el mejor de nuestra pila: +14 % en generación a c1 y +28 % a c4 frente a k=1, a cambio de una 1.ª respuesta algo más lenta
con 64K en caché (1,25 s frente a 0,97). k=3 y k=4 no mejoran: con nuestros kernels cada token propuesto de más cuesta 5-9 ms y la
aceptación por posición (c4) cae deprisa: 0,68 · 0,41 · 0,22 · 0,10 (en su TP=2, 0,72 · 0,46 · 0,27 · 0,15). A ursu le sale a cuenta
k=4 porque su paso de borrador es mucho más barato. En decode v4 a c1, dos de las tres repeticiones de k=1 y k=2 coincidieron con
tráfico real: para c1 vale más llama-benchy.</p>
<p>Reutilización de la caché con 64K + 2 048 tokens nuevos en nuestro residente (k=1): reaprovecha
{f((res or {}).get("pasos", [{}, {}])[1].get("reutilizados_metricas"), 0)} y recalcula
{f((res or {}).get("pasos", [{}, {}])[1].get("recalculados"), 0)} tokens (su TP=2: 63 008 y 4 659).</p>

<h3>2 · Su checkpoint NVFP4 en nuestra pila TP=3</h3>
<ul>
<li><b>Conversión.</b> Su checkpoint (HF, revisión 16c9bd54, el de su receta TP=2) convertido a vista TP=3 en el x86: cabezas KV
6, GDN 24/72, shared expert 640→768, vocabulario 248 448 con divisor 384, tabla PLE en FP8 con escala global. 119 GiB; verificación
byte a byte contra un gather ingenuo. Segunda vista con los expertos enrutados rellenos 640→768 (ceros en pesos, escala 1,0), que es
lo que pide el MoE b12x.</li>
<li><b>b12x sin imagen nueva.</b> b12x 1.3.0 es Python puro y sus 11 dependencias ya están en la imagen oficial de vLLM que usamos;
se instala al arrancar desde una rueda en la caché compartida (llave <code>QWEN38_B12X</code> en el launch).</li>
<li><b>Vocabulario recortado del borrador, con castellano.</b> Portado su <code>VLLM_MTP_DRAFT_VOCAB</code> a nuestra vLLM, con una
lista propia de 131 072 ids que cubre el 99,99 % del castellano (la suya, 95,97 %); el GEMM del borrador baja un 47 % por rango.
Para TP=3 hubo que rellenar la cabeza a 131 073 filas. Aún sin medir en GPU.</li>
<li><b>A · NVFP4 + b12x en MoE y GEMM densos, k=2: no arranca.</b> Carga, elige el MoE B12X y calienta sus kernels, y al capturar
los grafos CUDA da un acceso ilegal a memoria. Como el error es asíncrono, salta en el allreduce que sigue a una capa GDN.</li>
<li><b>A1 · NVFP4 (expertos de 640 con EP=3) + b12x solo en los GEMM densos, k=2: arranca en 9 min y sirve.</b> Herramientas: sin
stream bien, en stream una de las dos vueltas no acabó en texto; /v1/messages y Claude Code, bien. Velocidad en las tablas de arriba.
{('Caché a 64K: reaprovecha ' + f(a1c['pasos'][1]['reutilizados_metricas'], 0) + ' y recalcula ' + f(a1c['pasos'][1]['recalculados'], 0) + ' tokens (1.ª respuesta ' + f(a1c['pasos'][1]['ttft_s'], 2) + ' s).') if a1c else 'Caché a 64K: en curso.'}</li>
<li><b>A1 frente a nuestro FP8.</b> En decode v4 A1 es lo más rápido que hemos medido en TP=3: c1 55,5 · c4 160 · c8 256 · c10 271
tok/s (FP8 con k=2: 44,8 · 145 · 239 · 277), y a c8-c10 ya pasa a su TP=2. En llama-benchy el cuadro es más mixto: c1 57 frente a 56,
c4 102 frente a 123, y con 64K en caché la 1.ª respuesta queda casi igual (1,36 s frente a 1,25 a c1; 5,9 frente a 5,8 a c8), muy
lejos de los 2,2 y 10,6 s de su receta. El bloque de caché sigue siendo de 832 tokens porque el estado SSM sigue en BF16.</li>
<li><b>Conclusión por ahora:</b> los GEMM densos b12x funcionan en TP=3; el fallo está en el MoE b12x sin EP (o en el relleno de los
expertos a 768). Se está aislando el kernel exacto (arranque con <code>CUDA_LAUNCH_BLOCKING=1</code> y un micro-test de la capa MoE en
un solo Spark) comparando nuestro <code>fused_moe/b12x.py</code> con el del fork de ursu, donde funciona.</li>
</ul>

<h3>3 · Qué falta</h3>
<ol>
<li>Arreglar el MoE b12x en TP=3 y medir la configuración completa (A): velocidad, caché a 64K y calidad en 7 suites (la NVFP4 puede
perder nota; en DGX-847 empató, pero el castellano no se midió aparte).</li>
<li>Con el vocabulario recortado, repetir k=2, 3 y 4: si con el borrador barato k=4 pasa a ganar, como en su receta.</li>
<li>Fijar la mejor combinación en producción por PR.</li>
</ol>
"""
(D / "dgx852.html").write_text(out)
print(D / "dgx852.html", len(out))
