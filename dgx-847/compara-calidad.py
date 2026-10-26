"""DGX-847: calidad de las recetas de ursu (2×, 1×) frente a nuestro qwen38 TP=3 (DGX-817), pareada por ítem con compare()
de tools/qwen38_quality/report.py (10.000 remuestreos, semilla 7, como DGX-817/818). Usa los brazos que tengan items.
Uso: python3 -I compara-calidad.py   (escribe calidad-vs-tp3.json y .md junto al script)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bench-results/dgx-818/src/tools/qwen38_quality"))
import report
B = Path.home() / "bench-results"; D = B / "dgx-847"
ROOTS = {"tp3": B / "dgx-817/tp3-quality", "2x": D / "2x/quality", "1x": D / "1x/quality"}
arms = {}
for name, root in ROOTS.items():
    for p in sorted(root.glob("items-*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); arms.setdefault(r["suite"], {}).setdefault(name, {})[r["item_id"]] = bool(r["ok"])
out, md = {}, ["| suite | TP3 | 2× | 1× | 2×−TP3 (IC95) | 1×−TP3 (IC95) |", "|---|---|---|---|---|---|"]
for s, a in sorted(arms.items()):
    out[s] = {"k": {n: [sum(v.values()), len(v)] for n, v in a.items()}}
    cel = lambda n: f"{out[s]['k'][n][0]}/{out[s]['k'][n][1]}" if n in a else "—"
    dl = []
    for n in ("2x", "1x"):
        if n in a and "tp3" in a and set(a[n]) & set(a["tp3"]):
            c = report.compare(a, (n,), ("tp3",), floor_pairs=())
            out[s][f"{n}_vs_tp3"] = {"n": c["n"], "delta": c["delta"], "ci95": c["ci95"], "excluye0": bool(c["ci95"][0] > 0 or c["ci95"][1] < 0)}
            dl.append(f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{' *' if out[s][f'{n}_vs_tp3']['excluye0'] else ''}")
        else:
            dl.append("—")
    md.append(f"| {s} | {cel('tp3')} | {cel('2x')} | {cel('1x')} | {dl[0]} | {dl[1]} |")
(D / "calidad-vs-tp3.json").write_text(json.dumps(out, indent=1))
(D / "calidad-vs-tp3.md").write_text("\n".join(md + ["", "* = el IC95 excluye el 0 (diferencia real)."]) + "\n")
print("\n".join(md))
