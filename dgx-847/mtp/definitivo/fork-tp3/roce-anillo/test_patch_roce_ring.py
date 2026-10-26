#!/usr/bin/env python3
"""DGX-852 · Pruebas de CPU del parche B12X_ROCE_RING (roce-anillo).

Cubre:
  1. anclas: aplicar, idempotencia, dry-run y fail-closed (ancla corrupta -> 3,
     sin escribir NADA), sobre una copia de los ficheros reales del fork.
  2. `py_compile` de los dos .py parcheados.
  3. El .c parcheado con `gcc -fsyntax-only` contra el DOBLE de libibverbs de
     stub_verbs/ (en esta x86 no hay cabeceras de libibverbs reales: el doble
     prueba lógica, la compilación de la ABI real ocurre en la imagen).
  4. Camino C identidad vs original: banco harness_roce_post.c enlazado contra
     el .c ORIGINAL y contra el PARCHeADO; el stream de ibv_modify_qp(RTR) e
     ibv_post_send debe ser BYTE A BYTE idéntico sin B12X_ROCE_PEER_HCAS.
  5. Camino C enrutado: con las rutas del triángulo, cada par escribe payload y
     flags por la puerta correcta, con el rkey de la HCA remota, y los flags
     cubren todos los slots del receptor.
  6. Generador roce_ring_map.py: salida esperada por rango, simetría, mitad B,
     sha256 de la copia congelada de ring.conf y consistencia con las rutas
     que usa el banco.
  7. Parser y validación Python (parse_peer_hcas / validate_peer_routes),
     extraídos del módulo parcheado y ejecutados sin torch.

  python3 -I test_patch_roce_ring.py
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK_SRC = Path(
    "/home/user/bench-results/dgx-847/fork-src/usr/local/lib/python3.12/dist-packages"
)
REL = [
    "b12x/comm/roce/_roce_proxy.c",
    "b12x/comm/roce/_proxy.py",
    "b12x/comm/roce/roce_oneshot.py",
]
PATCH = HERE / "patch_roce_ring.py"
GEN = HERE / "roce_ring_map.py"
STUB = HERE / "stub_verbs"
RING_SHA = "14fd293043c1fecb016df85470b1d91cc5672b202bc28ed03db71c5ad2252c9f"

MARKERS = [
    ("_roce_proxy.c", ["B12X_ROCE_RING_ABI", "B12X_ROCE_RING_CONNFIELD",
                       "B12X_ROCE_RING_ROUTE", "B12X_ROCE_RING_INIT",
                       "B12X_ROCE_RING_SETFN", "B12X_ROCE_RING_CONNECT",
                       "B12X_ROCE_RING_CONNECTLOOP", "B12X_ROCE_RING_POST"]),
    ("_proxy.py", ["B12X_ROCE_RING_ABI_PY", "B12X_ROCE_RING_BIND",
                   "B12X_ROCE_RING_ROUTES"]),
    ("roce_oneshot.py", ["B12X_ROCE_RING_FUNCS", "B12X_ROCE_RING_PARSE",
                         "B12X_ROCE_RING_CFG", "B12X_ROCE_RING_ROUTES_PY"]),
]

# Rutas del triángulo A por rango, sobre hca_names = [rocep1s0f0, rocep1s0f1]
# (las que emite el generador y las que enlaza el banco).
ROUTES_BY_RANK = {
    0: {1: (0, 0), 2: (1, 1)},
    1: {0: (0, 0), 2: (1, 0)},
    2: {0: (1, 1), 1: (0, 1)},
}

FAILURES: list[str] = []
PASSED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def make_root(tmp: Path) -> Path:
    root = tmp / "dist"
    for rel in REL:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FORK_SRC / rel, dst)
    return root


def run_patch(root: Path, gate: str = "1", extra: list[str] | None = None):
    env = dict(os.environ)
    if gate is None:
        env.pop("B12X_ROCE_RING", None)
    else:
        env["B12X_ROCE_RING"] = gate
    proc = subprocess.run(
        [sys.executable, "-I", str(PATCH), "--root", str(root), *(extra or [])],
        capture_output=True, text=True, env=env,
    )
    return proc


def digest(root: Path) -> dict[str, str]:
    return {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in REL}


# --------------------------------------------------------------------------
# 1-2: anclas, idempotencia, dry-run, fail-closed, py_compile
# --------------------------------------------------------------------------

def test_anchors() -> None:
    print("[anclas]")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = make_root(tmp)
        before = digest(root)

        proc = run_patch(root, gate=None)
        check("inerte sin B12X_ROCE_RING", proc.returncode == 0 and "inerte" in proc.stdout,
              proc.stdout + proc.stderr)
        check("inerte no escribe", digest(root) == before)

        proc = run_patch(root, gate="2")
        check("gate invalido -> exit 2", proc.returncode == 2, proc.stdout + proc.stderr)

        proc = run_patch(root)
        check("aplicar -> exit 0", proc.returncode == 0, proc.stdout + proc.stderr)
        check("aplicado (no ya aplicado)", "aplicado:" in proc.stdout and "ya aplicado" not in proc.stdout,
              proc.stdout)
        for fname, markers in MARKERS:
            text = (root / "b12x/comm/roce" / fname).read_text()
            missing = [m for m in markers if m not in text]
            check(f"marcas en {fname}", not missing, f"faltan {missing}")

        # no solo las marcas: el código funcional tiene que estar insertado
        # una vez (una marca ausente en el texto nuevo duplica la edición al
        # segundo arranque; una colisión de marcas la omite al primero)
        proxy_text = (root / "b12x/comm/roce/_proxy.py").read_text()
        oneshot_text = (root / "b12x/comm/roce/roce_oneshot.py").read_text()
        check("_proxy.py trae set_peer_hcas", proxy_text.count("def set_peer_hcas") == 1)
        check("oneshot lee B12X_ROCE_PEER_HCAS",
              oneshot_text.count('peer_raw = os.getenv("B12X_ROCE_PEER_HCAS", "")') == 1)
        check("oneshot define parse_peer_hcas una vez",
              oneshot_text.count("def parse_peer_hcas") == 1)
        check("oneshot define validate_peer_routes una vez",
              oneshot_text.count("def validate_peer_routes") == 1)
        check("oneshot enruta antes de conectar",
              oneshot_text.count("self._proxy.set_peer_hcas(routes)") == 1)
        c_text = (root / "b12x/comm/roce/_roce_proxy.c").read_text()
        check(".c define roce_set_peer_hca una vez",
              c_text.count("int roce_set_peer_hca(roce_ctx_t *c, int peer, int local_hca, int remote_hca) {") == 1)
        check(".c ABI 4", "#define ROCE_ABI_VERSION 4" in c_text)

        after_apply = digest(root)
        proc = run_patch(root)
        check("idempotencia -> exit 0", proc.returncode == 0, proc.stdout + proc.stderr)
        check("idempotencia: ya aplicado", "ya aplicado" in proc.stdout, proc.stdout)
        check("idempotencia: no reescribe", digest(root) == after_apply)

        # py_compile de los .py parcheados
        for fname in ("_proxy.py", "roce_oneshot.py"):
            path = root / "b12x/comm/roce" / fname
            try:
                compile(path.read_text(), str(path), "exec")
                check(f"py_compile {fname}", True)
            except SyntaxError as exc:
                check(f"py_compile {fname}", False, str(exc))

        # fail-closed: corromper un ancla del .c en una RAIZ VIRGEN -> exit 3 y
        # NADA escrito (ni el propio .c ni los .py, que sí casarían)
        root3 = make_root(tmp / "t3")
        corrupt = root3 / "b12x/comm/roce/_roce_proxy.c"
        text = corrupt.read_text()
        corrupt.write_text(text.replace(
            "    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS];\n",
            "    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS]; // movido\n"
            "    uint32_t extra_field;\n",
        ))
        corrupted = digest(root3)
        proc = run_patch(root3)
        check("ancla corrupta -> exit 3", proc.returncode == 3, proc.stdout + proc.stderr)
        check("fail-closed: no escribe nada", digest(root3) == corrupted,
              "algún fichero se escribió pese al ancla rota")

        # dry-run sobre una raíz virgen
        root2 = make_root(tmp / "t2")
        before2 = digest(root2)
        proc = run_patch(root2, extra=["--dry-run"])
        check("dry-run -> exit 0", proc.returncode == 0, proc.stdout + proc.stderr)
        check("dry-run no escribe", digest(root2) == before2)


# --------------------------------------------------------------------------
# 3-5: compilación del .c y comportamiento contra el doble de verbs
# --------------------------------------------------------------------------

def gcc_build(proxy_c: Path, out: Path, with_set_peer: bool) -> str | None:
    cmd = [
        "cc", "-std=gnu11", "-O1", "-I", str(STUB),
        "-o", str(out),
        str(STUB / "harness_roce_post.c"), str(proxy_c), str(STUB / "verbs_stub.c"),
        "-lpthread",
    ]
    if with_set_peer:
        cmd.insert(1, "-DROCE_HAVE_SET_PEER")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stdout + proc.stderr
    return None


def test_c_builds(tmp: Path, patched_c: Path) -> None:
    print("[.c con doble de libibverbs]")
    has_real = Path("/usr/include/infiniband/verbs.h").is_file()
    check("cabeceras reales de libibverbs presentes (se usa el doble)", not has_real,
          "hay /usr/include/infiniband/verbs.h; el doble sigue valiendo")
    for name, src in (("original", FORK_SRC / "b12x/comm/roce/_roce_proxy.c"),
                      ("parcheado", patched_c)):
        proc = subprocess.run(
            ["gcc", "-fsyntax-only", "-std=gnu11", "-I", str(STUB), str(src)],
            capture_output=True, text=True,
        )
        check(f"gcc -fsyntax-only {name}", proc.returncode == 0, proc.stderr)

    orig_bin = tmp / "harness_orig"
    patch_bin = tmp / "harness_patch"
    err = gcc_build(FORK_SRC / "b12x/comm/roce/_roce_proxy.c", orig_bin, False)
    check("enlaza banco con .c original", err is None, err or "")
    err = gcc_build(patched_c, patch_bin, True)
    check("enlaza banco con .c parcheado", err is None, err or "")
    if err is not None:
        return

    def run(binary: Path, mode: str) -> list[str]:
        proc = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"{binary.name} {mode}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        return proc.stdout.splitlines()

    # identidad: el stream RTR/WR/STATS del parcheado == el del original
    lines_orig = [l for l in run(orig_bin, "identity") if not l.startswith("ABI ")]
    lines_patch = [l for l in run(patch_bin, "identity") if not l.startswith("ABI ")]
    check("identidad: stream RTR/WR identico al original", lines_orig == lines_patch,
          "\n".join(f"orig: {a}\npatch:{b}" for a, b in zip(lines_orig, lines_patch) if a != b)[:800]
          or f"longitudes {len(lines_orig)} vs {len(lines_patch)}")
    wrs = [l for l in lines_orig if l.startswith("WR ")]
    check("identidad: 8 WR (2 pares x 2 HCAs, payload+flag)", len(wrs) == 8, str(len(wrs)))
    check("identidad: 4 terminaciones senalizadas",
          any(l.startswith("STATS") and "writes=4" in l for l in lines_orig),
          str([l for l in lines_orig if l.startswith("STATS")]))

    # enrutado: rutas del triangulo A desde el rango 0
    routed = run(patch_bin, "routed")
    check("routed: PASS del banco", any(l == "ROUTED PASS" for l in routed), str(routed[-1]))
    layout = next(l for l in routed if l.startswith("LAYOUT "))
    lay = dict(re.findall(r"(\w+)=(\d+)", layout))
    total, recv_off, flag_off = int(lay["total"]), int(lay["recv"]), int(lay["flag"])
    slot_bytes = 2 * 1024 * 1024
    wrs = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in routed if l.startswith("WR ")]
    check("routed: 6 WR (payload + 2 flags por par)", len(wrs) == 6, str(len(wrs)))

    def expect(peer: int, lh: int, rh: int):
        # el doble de verbs imprime las direcciones relativas en hex (%llx)
        qpn = 1000 + lh * 100 + (peer - 1)  # rank0: seq = peer-1
        rkey = f"0x{0x1000 + peer * 4 + rh:x}"
        base = peer * total
        payload = f"+{base + recv_off + slot_bytes:x}"  # slot = seq&1 = 1
        flag0 = f"+{base + flag_off + (1 * 2 + 0) * 128:x}"
        flag1 = f"+{base + flag_off + (1 * 2 + 1) * 128:x}"
        return qpn, rkey, payload, flag0, flag1

    for peer, (lh, rh) in ROUTES_BY_RANK[0].items():
        qpn, rkey, payload, flag0, flag1 = expect(peer, lh, rh)
        sel = [w for w in wrs if int(w["qpn"]) == qpn]
        check(f"routed: par peer{peer} usa solo qpn={qpn} (h{lh})", len(sel) == 3, str(sel))
        data = [w for w in sel if int(w["len"]) == 4096]
        check(f"routed: peer{peer} payload 4096 por qpn {qpn}", len(data) == 1, str(sel))
        if data:
            check(f"routed: peer{peer} raddr payload", data[0]["raddr"] == payload,
                  f"{data[0]['raddr']} != {payload}")
            check(f"routed: peer{peer} rkey de la HCA remota h{rh}", data[0]["rkey"] == rkey,
                  f"{data[0]['rkey']} != {rkey}")
            check(f"routed: peer{peer} payload sin senalizar", data[0]["flags"] == "0",
                  data[0]["flags"])
        flags = sorted((w for w in sel if int(w["len"]) == 4),
                       key=lambda w: int(w["raddr"].lstrip("+"), 16))
        check(f"routed: peer{peer} flags cubren los 2 slots del receptor",
              len(flags) == 2 and [f["raddr"] for f in flags] == [flag0, flag1],
              str([f.get("raddr") for f in flags]))
        check(f"routed: peer{peer} ultimo flag senalizado",
              len(flags) == 2 and flags[-1]["flags"] == "6" and flags[0]["flags"] == "4",
              str([f.get("flags") for f in flags]))
    other = [w for w in wrs if int(w["qpn"]) in (1001, 1100)]
    check("routed: ningun WR por las puertas equivocadas", not other, str(other))
    check("routed: 2 terminaciones (un flag senal por par)",
          any(l.startswith("STATS") and "writes=2" in l for l in routed),
          str([l for l in routed if l.startswith("STATS")]))


# --------------------------------------------------------------------------
# 6: generador del mapa
# --------------------------------------------------------------------------

def test_generator() -> None:
    print("[generador]")
    proc = subprocess.run([sys.executable, "-I", str(GEN), "--all"],
                          capture_output=True, text=True)
    check("generador --all -> exit 0", proc.returncode == 0, proc.stderr)
    expected = {
        0: "1:rocep1s0f0:rocep1s0f0,2:rocep1s0f1:rocep1s0f1",
        1: "0:rocep1s0f0:rocep1s0f0,2:rocep1s0f1:rocep1s0f0",
        2: "0:rocep1s0f1:rocep1s0f1,1:rocep1s0f0:rocep1s0f1",
    }
    got = {}
    rank = None
    for line in proc.stdout.splitlines():
        if line.startswith("# rango"):
            rank = int(line.split()[2])
        elif line.startswith("B12X_ROCE_PEER_HCAS="):
            got[rank] = line.split("=", 1)[1]
    check("mapas por rango esperados", got == expected, f"{got}")

    # consistencia con las rutas que enlaza el banco (nombres -> indices)
    names = ["rocep1s0f0", "rocep1s0f1"]
    as_idx = {}
    for rank, raw in expected.items():
        as_idx[rank] = {
            int(p): (names.index(l), names.index(r))
            for p, l, r in (item.split(":") for item in raw.split(","))
        }
    check("generador == rutas del banco", as_idx == ROUTES_BY_RANK, f"{as_idx}")

    proc = subprocess.run([sys.executable, "-I", str(GEN), "--check"],
                          capture_output=True, text=True)
    check("simetria A", proc.returncode == 0 and "SIMETRIA ok" in proc.stdout, proc.stdout + proc.stderr)
    proc = subprocess.run([sys.executable, "-I", str(GEN), "--check", "--half", "B"],
                          capture_output=True, text=True)
    check("simetria B", proc.returncode == 0, proc.stdout + proc.stderr)
    proc = subprocess.run([sys.executable, "-I", str(GEN), "--host", "dgx2", "--half", "B"],
                          capture_output=True, text=True)
    check("mitad B usa roceP2p1s0f0/1", "roceP2p1s0f0" in proc.stdout, proc.stdout)
    proc = subprocess.run([sys.executable, "-I", str(GEN), "--host", "noexiste"],
                          capture_output=True, text=True)
    check("host desconocido falla", proc.returncode != 0, proc.stdout + proc.stderr)

    body = Path(HERE / "ring.conf").read_text().splitlines(keepends=True)
    body_sha = hashlib.sha256("".join(body[4:]).encode()).hexdigest()
    check("ring.conf congelado: sha256 del cuerpo", body_sha == RING_SHA, body_sha)


# --------------------------------------------------------------------------
# 7: parser y validacion Python (extraidos del modulo parcheado, sin torch)
# --------------------------------------------------------------------------

def extract_funcs(patched_oneshot: Path):
    text = patched_oneshot.read_text()
    start = text.index("def parse_peer_hcas(")
    end = text.index("def is_supported(")
    ns: dict = {"Sequence": __import__("typing").Sequence}
    exec(compile(text[start:end], "extracted", "exec"), ns)
    return ns["parse_peer_hcas"], ns["validate_peer_routes"]


def test_python_funcs(root: Path) -> None:
    print("[python]")
    oneshot = root / "b12x/comm/roce/roce_oneshot.py"
    check("ancla TP=2 intacta (connect sigue igual)",
          oneshot.read_text().count("self._proxy.connect([s[1] for s in statuses])") == 1)
    parse, validate = extract_funcs(oneshot)

    check("parse vacio", parse("") == {})
    check("parse bueno",
          parse("1:rocep1s0f0:rocep1s0f0, 2:rocep1s0f1:rocep1s0f0") ==
          {1: ("rocep1s0f0", "rocep1s0f0"), 2: ("rocep1s0f1", "rocep1s0f0")})
    for bad in ("1:rocep1s0f0", "a:x:y", "1::y", "1:x:y,1:z:w"):
        try:
            parse(bad)
            check(f"parse rechaza {bad!r}", False, "no levanto")
        except ValueError:
            check(f"parse rechaza {bad!r}", True)

    hcas = ["rocep1s0f0", "rocep1s0f1"]
    ann = [
        {"hca_names": hcas, "peer_hcas": {1: ["rocep1s0f0", "rocep1s0f0"],
                                          2: ["rocep1s0f1", "rocep1s0f1"]}},
        {"hca_names": hcas, "peer_hcas": {0: ["rocep1s0f0", "rocep1s0f0"],
                                          2: ["rocep1s0f1", "rocep1s0f0"]}},
        {"hca_names": hcas, "peer_hcas": {0: ["rocep1s0f1", "rocep1s0f1"],
                                          1: ["rocep1s0f0", "rocep1s0f1"]}},
    ]
    routes = validate(1, 3, hcas, {0: ("rocep1s0f0", "rocep1s0f0"),
                                   2: ("rocep1s0f1", "rocep1s0f0")}, ann)
    check("validate triangulo -> indices", routes == {0: (0, 0), 2: (1, 0)}, str(routes))
    cases = {
        "asimetrico": (1, 3, hcas, {0: ("rocep1s0f0", "rocep1s0f1"),
                                    2: ("rocep1s0f1", "rocep1s0f0")}, ann),
        "nombre inexistente": (1, 3, hcas, {0: ("rocep1s0f9", "rocep1s0f0"),
                                            2: ("rocep1s0f1", "rocep1s0f0")}, ann),
        "falta un par": (1, 3, hcas, {0: ("rocep1s0f0", "rocep1s0f0")}, ann),
        "remota ajena": (1, 3, hcas, {0: ("rocep1s0f0", "rocep1s0f0"),
                                      2: ("rocep1s0f1", "rocep9s0f0")}, ann),
    }
    for name, args in cases.items():
        try:
            validate(*args)
            check(f"validate rechaza {name}", False, "no levanto")
        except RuntimeError:
            check(f"validate rechaza {name}", True)


def main() -> int:
    if not FORK_SRC.is_dir():
        print(f"ABORTA: falta la copia del fork en {FORK_SRC}")
        return 2
    test_anchors()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = make_root(tmp)
        proc = run_patch(root)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        test_c_builds(tmp, root / "b12x/comm/roce/_roce_proxy.c")
        test_generator()
        test_python_funcs(root)
    print(f"\n{len(PASSED)} ok, {len(FAILURES)} fallos")
    for f in FAILURES:
        print(f"  FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
