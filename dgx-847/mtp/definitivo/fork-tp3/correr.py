#!/usr/bin/env python3
"""Lanza el conversor en dgx1 por el camino NUMPY, no por torch (DGX-852).

En el host de dgx1 hay torch 2.10 (cpu) instalado y `conv780`/el conversor lo importarían si se le
deja: con `sys.modules["torch"] = None` cualquier `import torch` falla y se toma la rama numpy, que
es como se construyeron `qwen38fn-mtp-ursu-nvfp4-tp12` y la e768. No usamos `python3 -I` porque
numpy vive en el site de usuario (`/home/dibanerz/.local/lib/python3.12/site-packages`), que `-I`
desactiva. Los argumentos van tal cual al conversor (argv[0] es el launcher).
"""
import runpy
import sys
from pathlib import Path

sys.modules["torch"] = None
sys.argv = [str(Path(__file__).with_name("convert-ursu-nvfp4-tp3.py"))] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
