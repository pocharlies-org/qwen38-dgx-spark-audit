#!/usr/bin/env python3
"""DGX-852: comprueba los Requires-Dist de la rueda b12x 1.3.0 contra lo que trae la imagen.

Misma logica que el paso 2 del bloque de launch.sh (packaging.Requirement + specifier), pero con
la lista de distribuciones leida del snapshot de containerd del nodo (imagen-distpackages.txt),
para poder evaluarla en CPU, sin GPU y sin levantar el contenedor.

    python3 -I test-dependencias.py <rueda.whl> <imagen-distpackages.txt>
"""

import re
import sys
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

DIST_RE = re.compile(r"^(?P<name>[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)-(?P<ver>\d[^_]*)\.dist-info$")


def instalados(listado: Path) -> dict[str, Version]:
    """nombre-normalizado -> version, a partir de los directorios *.dist-info."""
    out: dict[str, Version] = {}
    for linea in listado.read_text().splitlines():
        nombre = linea.strip().rstrip("/")
        m = DIST_RE.match(nombre)
        if not m:
            continue
        clave = m.group("name").lower().replace("_", "-")
        # La misma distribucion puede aparecer en dos snapshots: la version mas alta gana.
        previo = out.get(clave)
        actual = Version(m.group("ver"))
        if previo is None or actual > previo:
            out[clave] = actual
    return out


def main() -> int:
    rueda, listado = Path(sys.argv[1]), Path(sys.argv[2])
    with zipfile.ZipFile(rueda) as z:
        meta = z.read("b12x-1.3.0.dist-info/METADATA").decode()
    exige = [Requirement(l.split(":", 1)[1].strip())
             for l in meta.splitlines() if l.startswith("Requires-Dist:")]
    hay = instalados(listado)
    print(f"rueda {rueda.name}: {len(exige)} Requires-Dist;  imagen: {len(hay)} distribuciones")
    mal, val = 0, 0
    for req in exige:
        if req.marker and not req.marker.evaluate({"extra": ""}):
            print(f"  (extra)  {req}")
            continue
        val += 1
        version = hay.get(req.name.lower().replace("_", "-"))
        if version is None:
            print(f"  FALTA    {req}  -> AUSENTE en la imagen")
            mal += 1
        elif req.specifier and not req.specifier.contains(version, prereleases=True):
            print(f"  NO CUMPLE {req}  -> instalado {version}")
            mal += 1
        else:
            print(f"  ok       {req}  -> {version}")
    print(f"\n{val - mal}/{val} cumplidas con lo que ya trae la imagen")
    return 0 if mal == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
