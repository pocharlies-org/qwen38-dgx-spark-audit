#!/usr/bin/env python3
"""DGX-852 · QWEN_GDN_MIX: decode GDN b12x con estado BF16 + prefill triton (A/B)
o prefill b12x en BF16 (B2).

Se ejecuta al arrancar el contenedor, ANTES de `vllm serve`, sobre los
ficheros instalados en /usr/local/lib/python3.12/dist-packages (igual que
nuestro launch-ursu-b12x.sh). No toca el checkpoint ni el cluster.

Modos (variable de entorno QWEN_GDN_MIX):
  (vacio)  -> no hace nada, exit 0. La imagen queda tal cual (receta de ursu
              o E2, segun EXTRA_ARGS).
  A        -> decode b12x + prefill triton, SIN registros compactos ni
              checkpoints diferidos. Relaja solo la politica de seleccion de
              backend de vLLM. El estado BF16 lo acepta el decode b12x de por
              si (b12x/sequence/gdn_decode/_impl.py:102); lo unico que lo
              impedia era la politica de vLLM
              (qwen_gdn_linear_attn.py:_resolve_gdn_backend_selection).
  B        -> A + registros compactos + checkpoints diferidos, por el camino
              B1 (prefill triton, registros EXTERNOS en el side buffer FP32 de
              cada capa). Ademas relaja:
                - vLLM refuse_reasons (gdn_deferred_commit.py) para prefill
                  distinto de b12x CUANDO compact esta activo (no hay bloque de
                  registro en el pool que el prefill triton pueda leer mal);
                - los tres guardas del kernel b12x de decode que exigen estado
                  FP32 para checkpoints diferidos, PERO solo con
                  external_records=True (los deltas viven en el buffer FP32
                  aparte; la base puede ser BF16: el kernel sube la base a FP32
                  antes de reproducirla).
  B2       -> decode b12x + prefill b12x, CON registros compactos + checkpoints
              diferidos, y el estado SSM en BF16. Es la unica B servible: con
              prefill y decode b12x existe la ruta mixta `_forward_core_b12x_mixed`
              (gdn_attn.py:203-212 construye `_b12x_mixed` solo si prefill==b12x)
              y los lotes mixtos van por el camino disenado, no por triton, asi
              que NO aborta a c2+ como B1. B2 lleva las relajaciones de DECODE
              del modo B (ediciones 4,5,6: Caps + init de los dos kernels de
              replay/commit, condicionadas a external_records) mas la mitad de
              PREFILL: que el kernel CuTe de prefill b12x cargue el estado BF16,
              acumule en FP32 y guarde BF16. NO lleva la relajacion del selector
              (edicion 1) ni la de refuse_reasons (ediciones 2,3): con prefill y
              decode b12x la politica original ya lo permite y `refuse_reasons`
              no añade motivo (prefill_backend == "b12x").

              Lo que toca B2 en b12x (todo menos las guardas es trabajo de
              kernel real, no un `if`):
                - Caps de prefill (`gdn_prefill/_impl.py`) y tuning
                  (`gdn_prefill/_tuning.py`): aceptan state_dtype bfloat16.
                - El stub de preparacion (`_shared/delta_prefill/preparation.py`)
                  creaba el recurrent_state de metadatos SIEMPRE en FP32; con la
                  AOT (`VLLM_USE_AOT_COMPILE=1`) eso compilaria un kernel FP32
                  que en runtime no casa con el binding BF16. Se deriva del
                  state_dtype del query (KDA sigue FP32, su Caps no lo cambia).
                - El kernel de recurrencia (`_shared/delta_prefill/_cute_kernels.py`):
                  recibe `state_type`, el puntero del estado pasa de Float32 a
                  state_type, `_store_state` convierte el acumulador FP32 al tipo
                  del estado (espejo exacto del decode: carga con Float32(...),
                  guarda con self.state_type(...)), y la clave de cache incluye
                  el dtype para no reutilizar un kernel FP32 con binding BF16.
                - Los kernels auxiliares del chunk-parallel
                  (`gdn_prefill/_parallel_kernels.py`): el commit (`_CommitKernel`)
                  lee/escribe el estado real (final + checkpoints) y el boundary
                  (`_BoundaryKernel`) lo lee; ambos punteros pasan a state_type y
                  el commit convierte al guardar. El pool interno (`_parallel.py:57`)
                  y los registros compactos siguen FP32: son scratch, no el estado.

              Verificacion en GPU pendiente: que la numerica BF16 del prefill
              quede dentro del redondeo BF16 del estado (test_b2_gpu.py). El
              layout se comprueba al bindear (require_paged_recurrent_state), asi
              que un desajuste aborta, no corrompe.

El modo B es EXPERIMENTAL y tiene un LIMITE que no es de este parche: con
prefill triton los lotes MIXTOS (alguien prefillando y otros decodificando con
especulativos) no pueden ir por b12x -- `_can_use_b12x_gdn_decode` exige
num_prefills==0 (qwen_gdn_linear_attn.py:2899-2917) y el camino b12x_mixed solo
existe con prefill b12x (gdn_attn.py:203-212). Caen al triton, que lee la base
SIN los registros pendientes. El fork lo sabe y aborta
(qwen_gdn_linear_attn.py:3010-3014). Por eso B solo es util a c1; B2 es la
version servible.

Idempotente (marca por edicion), anclado con count(old)==1, FAIL-CLOSED: si se
pide A, B o B2 y un ancla no casa exactamente una vez, sale con 3 y un mensaje
claro y no escribe nada.

Uso:
  QWEN_GDN_MIX=A  python3 -I patch_gdn_mix.py                 # en la imagen
  QWEN_GDN_MIX=B2 python3 -I patch_gdn_mix.py --root <dist-packages>   # test CPU
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARK = "QWEN_GDN_MIX"

# ---------------------------------------------------------------------------
# Anclas. Cada edicion: (ruta-relativa, viejo, nuevo, marca-unica, modos).
#   modos -> tupla de valores de QWEN_GDN_MIX para los que se aplica.
#   ("A","B")    -> relajacion de seleccion (A y B; B2 no la necesita).
#   ("B",)       -> relajaciones propias del camino B1 (prefill triton + diferidos).
#   ("B","B2")   -> relajaciones de DECODE (base BF16 + registros FP32 externos).
#   ("B2",)      -> mitad de PREFILL: el kernel b12x de prefill en BF16.
# `nuevo` incluye `marca` para que la idempotencia se detecte buscandola.
# ---------------------------------------------------------------------------

GDN_LAYER = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
DEFERRED = "vllm/v1/worker/gdn_deferred_commit.py"
B12X_DEC_IMPL = "b12x/sequence/gdn_decode/_impl.py"
B12X_DEC_CUTE = "b12x/sequence/gdn_decode/_cute_kernels.py"
B12X_DEC_TUNING = "b12x/sequence/gdn_decode/_tuning.py"
B12X_PRE_IMPL = "b12x/sequence/gdn_prefill/_impl.py"
B12X_PRE_TUNING = "b12x/sequence/gdn_prefill/_tuning.py"
B12X_PRE_PARALLEL = "b12x/sequence/gdn_prefill/_parallel_kernels.py"
B12X_PARALLEL = "b12x/sequence/gdn_prefill/_parallel.py"
DELTA_CUTE = "b12x/sequence/_shared/delta_prefill/_cute_kernels.py"
DELTA_PREP = "b12x/sequence/_shared/delta_prefill/preparation.py"

EDITS: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    # (1) Seleccion de backend: permitir decode b12x + prefill no-b12x (A y B).
    (
        GDN_LAYER,
        '    if prefill == "b12x" or decode == "b12x":\n'
        '        if prefill not in ("auto", "b12x") or decode not in (None, "b12x"):\n'
        "            raise ValueError(\n"
        '                "b12x GDN prefill and decode must be selected together; "\n'
        '                f"got prefill={prefill!r}, decode={decode!r}. "\n'
        '                "Remove the conflicting GDN backend override."\n'
        "            )\n"
        '        return "b12x", "b12x", explicitly_configured\n',
        '    if prefill == "b12x" or decode == "b12x":\n'
        '        if prefill not in ("auto", "b12x") or decode not in (None, "b12x"):\n'
        f"            # {MARK}_SELECTION (DGX-852): decode b12x con prefill no-b12x.\n"
        '            _mix = os.environ.get("QWEN_GDN_MIX", "").strip().upper()\n'
        "            if (\n"
        '                _mix in ("A", "B")\n'
        '                and decode == "b12x"\n'
        '                and prefill in ("triton", "flashinfer", "cutedsl")\n'
        "            ):\n"
        "                pass\n"
        "            else:\n"
        "                raise ValueError(\n"
        '                    "b12x GDN prefill and decode must be selected together; "\n'
        '                    f"got prefill={prefill!r}, decode={decode!r}. "\n'
        '                    "Remove the conflicting GDN backend override."\n'
        "                )\n"
        "        else:\n"
        '            return "b12x", "b12x", explicitly_configured\n',
        f"{MARK}_SELECTION",
        ("A", "B"),
    ),
    # (2) gdn_deferred_commit.py: anadir `import os` (no lo tiene). Solo B1.
    (
        DEFERRED,
        "import torch\nimport triton\nimport triton.language as tl\n",
        f"# {MARK}_IMPORTOS (DGX-852): refuse_reasons lee QWEN_GDN_MIX.\n"
        "import os\nimport torch\nimport triton\nimport triton.language as tl\n",
        f"{MARK}_IMPORTOS",
        ("B",),
    ),
    # (3) refuse_reasons: aceptar prefill != b12x cuando compact esta activo. Solo B1.
    (
        DEFERRED,
        '    if prefill_backend != "b12x":\n'
        "        reasons.append(\n"
        '            f"requires the b12x GDN prefill backend (got {prefill_backend!r}); "\n'
        '            "another prefill kernel could read a record block as a state"\n'
        "        )\n",
        '    if prefill_backend != "b12x":\n'
        f"        # {MARK}_REFUSE (DGX-852): con COMPACT activo los registros por\n"
        "        # token viven en un side buffer FP32 por capa y el spec mamba suelta\n"
        "        # sus bloques especulativos (num_speculative_blocks=0), asi que el\n"
        "        # pool no tiene bloque de registro que un prefill triton pueda leer\n"
        "        # mal. PERO la base del bloque en curso sigue estando incompleta entre\n"
        "        # fronteras: un lote mixto que caiga al triton perderia los registros\n"
        "        # pendientes, y el fork lo aborta mas abajo (guard compact+spec en\n"
        "        # qwen_gdn_linear_attn.py). AQUI solo se levanta el motivo de prefill;\n"
        "        # B queda valido a c1 y aborta a c2+, que es lo que mide PLAN-GPU.md.\n"
        '        _mix = os.environ.get("QWEN_GDN_MIX", "").strip().upper()\n'
        "        if not (\n"
        '            _mix == "B"\n'
        '            and decode_kernel == "b12x"\n'
        "            and compact_requested()\n"
        "        ):\n"
        "            reasons.append(\n"
        '                f"requires the b12x GDN prefill backend (got {prefill_backend!r}); "\n'
        '                "another prefill kernel could read a record block as a state"\n'
        "            )\n",
        f"{MARK}_REFUSE",
        ("B",),
    ),
    # (4) b12x decode Caps: permitir BF16 con external_records (B y B2).
    (
        B12X_DEC_IMPL,
        "            if self.state_dtype != torch.float32:\n"
        "                raise ValueError(\n"
        '                    "deferred checkpoints require an FP32 recurrent state, got "\n'
        '                    f"{self.state_dtype}"\n'
        "                )\n",
        "            if self.state_dtype != torch.float32 and not self.external_records:\n"
        f"                # {MARK}_CAPS (DGX-852): con registros externos (compact) los\n"
        "                # deltas van en un buffer FP32 aparte; solo la base se guarda en\n"
        "                # el dtype del estado, y el kernel la sube a FP32 antes de\n"
        "                # reproducir el prefijo. Base BF16: coherente por layout, SIN\n"
        "                # verificar en GPU (y no bit-identica al replay FP32).\n"
        "                raise ValueError(\n"
        '                    "deferred checkpoints require an FP32 recurrent state, got "\n'
        '                    f"{self.state_dtype}"\n'
        "                )\n",
        f"{MARK}_CAPS",
        ("B", "B2"),
    ),
    # (5) b12x cute decode kernel init: permitir BF16 con external_records (B y B2).
    (
        B12X_DEC_CUTE,
        "        if self.deferred_checkpoints:\n"
        "            if state_type is not Float32:\n"
        "                raise ValueError(\n"
        '                    "deferred GDN checkpoints require an FP32 recurrent state"\n'
        "                )\n"
        "            if null_state_index is not None:\n",
        "        if self.deferred_checkpoints:\n"
        "            if state_type is not Float32 and not self.external_records:\n"
        f"                # {MARK}_CUTEDEC (DGX-852): base BF16 permitida solo con\n"
        "                # registros externos (compact); los deltas quedan en FP32.\n"
        "                raise ValueError(\n"
        '                    "deferred GDN checkpoints require an FP32 recurrent state"\n'
        "                )\n"
        "            if null_state_index is not None:\n",
        f"{MARK}_CUTEDEC",
        ("B", "B2"),
    ),
    # (6) b12x cute commit kernel init: permitir BF16 con external_records (B y B2).
    (
        B12X_DEC_CUTE,
        "        self.value_heads = int(value_heads)\n"
        "        if state_type is not Float32:\n"
        "            raise ValueError(\n"
        '                "deferred GDN checkpoints require an FP32 recurrent state"\n'
        "            )\n"
        "        self.state_type = state_type\n",
        "        self.value_heads = int(value_heads)\n"
        "        if state_type is not Float32 and not external_records:\n"
        f"            # {MARK}_CUTECOMMIT (DGX-852): base BF16 permitida solo con\n"
        "            # registros externos; el commit sube la base a FP32 al reproducir.\n"
        "            raise ValueError(\n"
        '                "deferred GDN checkpoints require an FP32 recurrent state"\n'
        "            )\n"
        "        self.state_type = state_type\n",
        f"{MARK}_CUTECOMMIT",
        ("B", "B2"),
    ),
    # (6b) b12x decode: la validacion de la consulta de preparacion (B y B2). Sin ella
    # B2 no arranca: "deferred GDN checkpoints require Qwen heads, FP32 state" (14-10 12:15Z).
    (
        B12X_DEC_TUNING,
        "    if query.deferred_checkpoints and (\n"
        "        kda\n"
        "        or query.state_dtype != \"float32\"\n"
        "        or query.null_state_index is not None\n"
        "    ):\n",
        "    if query.deferred_checkpoints and (\n"
        "        kda\n"
        f"        # {MARK}_DECTUNE (DGX-852): base BF16 solo con registros externos FP32\n"
        "        or (query.state_dtype != \"float32\" and query.external_records is not True)\n"
        "        or query.null_state_index is not None\n"
        "    ):\n",
        f"{MARK}_DECTUNE",
        ("B", "B2"),
    ),
    # ------------------------------------------------------------------
    # B2: mitad de PREFILL b12x en BF16. Solo se aplica con QWEN_GDN_MIX=B2.
    # ------------------------------------------------------------------
    # (7) Caps de prefill: aceptar state_dtype bfloat16.
    (
        B12X_PRE_IMPL,
        "        if self.state_dtype != torch.float32:\n"
        '            raise ValueError("state_dtype must be torch.float32")\n',
        f"        # {MARK}_PREFCAPS (DGX-852 B2): el prefill b12x acepta estado BF16\n"
        "        # (base). Carga BF16, acumula en FP32 y guarda BF16; el pool interno\n"
        "        # de la ruta chunk-parallel sigue FP32 (es scratch, no el estado).\n"
        "        if self.state_dtype not in (torch.float32, torch.bfloat16):\n"
        '            raise ValueError("state_dtype must be torch.float32 or torch.bfloat16")\n',
        f"{MARK}_PREFCAPS",
        ("B2",),
    ),
    # (8) tuning de prefill: aceptar state_dtype bfloat16.
    (
        B12X_PRE_TUNING,
        '    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":\n'
        "        raise ValueError(\n"
        '            "sequence.gdn_prefill requires bfloat16 activations and float32 state, "\n'
        '            f"got {query.model_dtype}/{query.state_dtype}"\n'
        "        )\n",
        f"    # {MARK}_PREFTUNE (DGX-852 B2): estado BF16 admitido en el query.\n"
        '    if query.model_dtype != "bfloat16" or query.state_dtype not in ("float32", "bfloat16"):\n'
        "        raise ValueError(\n"
        '            "sequence.gdn_prefill requires bfloat16 activations and float32 or "\n'
        '            f"bfloat16 state, got {query.model_dtype}/{query.state_dtype}"\n'
        "        )\n",
        f"{MARK}_PREFTUNE",
        ("B2",),
    ),
    # (9) preparacion: el stub de metadatos del recurrent_state usa el dtype real.
    #     Sin esto, con AOT se compila un kernel FP32 que no casa con el binding BF16.
    (
        DELTA_PREP,
        "        dt_bias=tensor(getattr(torch, query.dt_bias_dtype)), recurrent_state=tensor(torch.float32),\n",
        f"        # {MARK}_PREFPREP (DGX-852 B2): el recurrent_state de metadatos\n"
        "        # refleja el state_dtype del query (KDA sigue float32: su Caps no lo\n"
        "        # cambia), para que la AOT compile el mismo dtype que el binding real.\n"
        "        dt_bias=tensor(getattr(torch, query.dt_bias_dtype)),\n"
        "        recurrent_state=tensor(getattr(torch, query.state_dtype)),\n",
        f"{MARK}_PREFPREP",
        ("B2",),
    ),
    # (9b) preparacion: el binding interno del chunk_parallel apunta al POOL FP32,
    #      no al estado real. Debe seguir declarando float32.
    (
        DELTA_PREP,
        '        inner_query = replace(query, state_indices_dtype="int32")\n',
        "        # " + f"{MARK}_PREFINNER (DGX-852 B2): en chunk_parallel el\n"
        "        # recurrent_state de los bindings internos es el pool de scratch,\n"
        "        # que _parallel.materialize fija en FP32. Si heredara el state_dtype\n"
        "        # BF16 del cap, el kernel de recurrencia interno se compilaria en\n"
        "        # BF16 y el launch lo rechazaria por clave distinta.\n"
        '        inner_query = replace(query, state_indices_dtype="int32",\n'
        '                              state_dtype="float32")\n',
        f"{MARK}_PREFINNER",
        ("B2",),
    ),
    # (9c) _parallel.materialize: el plan interno se binda contra el pool FP32,
    #      asi que sus Caps deben declarar float32 (si no, bind_tensors rechaza
    #      el pool por dtype en chunk_parallel).
    (
        B12X_PARALLEL,
        "    inner_caps = replace(caps, max_seqs=segments, max_state_slots=slots, null_state_index=0)\n",
        "    # " + f"{MARK}_PREFPOOL (DGX-852 B2): el pool de scratch de la ruta\n"
        "    # chunk_parallel es FP32 fijo (`regions` abajo); los caps del plan\n"
        "    # interno lo reflejan aunque el estado real del capa sea BF16.\n"
        "    inner_caps = replace(caps, max_seqs=segments, max_state_slots=slots,\n"
        "                         null_state_index=0, state_dtype=torch.float32)\n",
        f"{MARK}_PREFPOOL",
        ("B2",),
    ),
    # (10) _RecurrenceKernel.__init__: anadir el parametro state_type.
    (
        DELTA_CUTE,
        "        max_sequence_tiles: int = 0,\n"
        "        is_gdn: bool = False,\n"
        "    ) -> None:\n",
        "        max_sequence_tiles: int = 0,\n"
        "        is_gdn: bool = False,\n"
        f"        state_type: type[cutlass.Numeric] = Float32,  # {MARK}_PREFRT (DGX-852 B2)\n"
        "    ) -> None:\n",
        f"{MARK}_PREFRT",
        ("B2",),
    ),
    # (11) _RecurrenceKernel.__init__: guardar self.state_type.
    (
        DELTA_CUTE,
        "        self.index_type = index_type\n",
        f"        self.index_type = index_type\n"
        f"        self.state_type = state_type  # {MARK}_PREFRTSTORE (DGX-852 B2)\n",
        f"{MARK}_PREFRTSTORE",
        ("B2",),
    ),
    # (12) _store_state base: convertir el acumulador FP32 al tipo del estado.
    #      Espejo del decode (guarda con self.state_type); la carga ya ensancha
    #      con Float32(...), asi que BF16 -> FP32 -> BF16 queda consistente.
    (
        DELTA_CUTE,
        "            target[offset0] = acc[nb, 0]\n"
        "            target[offset0 + Int64(1)] = acc[nb, 1]\n"
        "            target[offset1] = acc[nb, 2]\n"
        "            target[offset1 + Int64(1)] = acc[nb, 3]\n",
        f"            # {MARK}_PREFSTORE (DGX-852 B2): guarda en el dtype del estado.\n"
        "            target[offset0] = self.state_type(acc[nb, 0])\n"
        "            target[offset0 + Int64(1)] = self.state_type(acc[nb, 1])\n"
        "            target[offset1] = self.state_type(acc[nb, 2])\n"
        "            target[offset1 + Int64(1)] = self.state_type(acc[nb, 3])\n",
        f"{MARK}_PREFSTORE",
        ("B2",),
    ),
    # (13) _recurrence_key: incluir el dtype del estado (cache/AOT no colisionan).
    (
        DELTA_CUTE,
        "        caps.checkpoint_export,\n"
        "        caps.null_state_index,\n"
        "        binding.initial_state_indices.dtype,\n"
        "    )\n",
        "        caps.checkpoint_export,\n"
        "        caps.null_state_index,\n"
        "        binding.initial_state_indices.dtype,\n"
        f"        binding.recurrent_state.dtype,  # {MARK}_PREFKEY (DGX-852 B2)\n"
        "    )\n",
        f"{MARK}_PREFKEY",
        ("B2",),
    ),
    # (14) _compile_recurrence: derivar state_type del binding.
    (
        DELTA_CUTE,
        "    index_type = _numeric_type(binding.initial_state_indices.dtype)\n",
        "    index_type = _numeric_type(binding.initial_state_indices.dtype)\n"
        f"    state_type = _numeric_type(binding.recurrent_state.dtype)  # {MARK}_PREFCT (DGX-852 B2)\n",
        f"{MARK}_PREFCT",
        ("B2",),
    ),
    # (15) _compile_recurrence: pasar state_type al kernel.
    (
        DELTA_CUTE,
        "        null_state_index=caps.null_state_index,\n"
        "        index_type=index_type,\n"
        "        summary_mode=summary_mode,\n",
        "        null_state_index=caps.null_state_index,\n"
        "        index_type=index_type,\n"
        f"        state_type=state_type,  # {MARK}_PREFCARG (DGX-852 B2)\n"
        "        summary_mode=summary_mode,\n",
        f"{MARK}_PREFCARG",
        ("B2",),
    ),
    # (16) _compile_recurrence: puntero falso del estado -> state_type.
    (
        DELTA_CUTE,
        "        _fake_pointer(cutlass.Int8),\n"
        "        _fake_pointer(Float32),\n"
        "        _fake_pointer(BFloat16),\n",
        "        _fake_pointer(cutlass.Int8),\n"
        f"        _fake_pointer(state_type),  # {MARK}_PREFCFAKE (DGX-852 B2)\n"
        "        _fake_pointer(BFloat16),\n",
        f"{MARK}_PREFCFAKE",
        ("B2",),
    ),
    # (17) _compile_recurrence: puntero real del estado en el launch -> state_type.
    (
        DELTA_CUTE,
        "            _pointer(active.ws.view(torch.int8), cutlass.Int8),\n"
        "            _pointer(active.recurrent_state, Float32),\n"
        "            _pointer(active.output, BFloat16),\n",
        "            _pointer(active.ws.view(torch.int8), cutlass.Int8),\n"
        f"            _pointer(active.recurrent_state, state_type),  # {MARK}_PREFCPTR (DGX-852 B2)\n"
        "            _pointer(active.output, BFloat16),\n",
        f"{MARK}_PREFCPTR",
        ("B2",),
    ),
    # (18) _CommitKernel.__init__: parametro self.state_type (escribe el estado real).
    (
        B12X_PRE_PARALLEL,
        "    def __init__(self, *, heads: int, max_seqs: int, max_segments: int,\n"
        "                 null_state_index: int | None, reuse_outputs: bool) -> None:\n"
        "        self.heads = heads\n",
        "    def __init__(self, *, heads: int, max_seqs: int, max_segments: int,\n"
        "                 null_state_index: int | None, reuse_outputs: bool,\n"
        f"                 state_type: type[cutlass.Numeric] = Float32) -> None:  # {MARK}_PREFCOMMIT (DGX-852 B2)\n"
        "        self.state_type = state_type\n"
        "        self.heads = heads\n",
        f"{MARK}_PREFCOMMIT",
        ("B2",),
    ),
    # (19) _CommitKernel: el estado final se guarda en el dtype del estado.
    (
        B12X_PRE_PARALLEL,
        "                    state[target * state_stride + head_base + element] = value\n",
        f"                    # {MARK}_PREFCOMMITSTORE (DGX-852 B2): base BF16 -> self.state_type.\n"
        "                    state[target * state_stride + head_base + element] = self.state_type(value)\n",
        f"{MARK}_PREFCOMMITSTORE",
        ("B2",),
    ),
    # (20) _CommitKernel: el checkpoint se guarda en el dtype del estado.
    (
        B12X_PRE_PARALLEL,
        "                        state[cp * state_stride + head_base + element] = Float32(pool[slot * pool_stride + head_base + element])\n",
        f"                        # {MARK}_PREFCOMMITCKPT (DGX-852 B2): checkpoint en el dtype del estado.\n"
        "                        state[cp * state_stride + head_base + element] = self.state_type(Float32(pool[slot * pool_stride + head_base + element]))\n",
        f"{MARK}_PREFCOMMITCKPT",
        ("B2",),
    ),
    # (21) compile_auxiliary: derivar state_type y anadirlo a la clave.
    (
        B12X_PRE_PARALLEL,
        "    caps = binding._state.caps\n"
        "    key = (binding.output.device.index, caps.heads, caps.max_seqs,\n"
        "           parallel._state.max_segments, parallel._state.segment_tokens,\n"
        "           binding._state.v_split, caps.null_state_index, binding.initial_state_indices.dtype,\n"
        "           parallel._state.reuse_outputs)\n",
        "    caps = binding._state.caps\n"
        f"    state_type = _numeric_type(binding.recurrent_state.dtype)  # {MARK}_PREFAXKEY (DGX-852 B2)\n"
        "    key = (binding.output.device.index, caps.heads, caps.max_seqs,\n"
        "           parallel._state.max_segments, parallel._state.segment_tokens,\n"
        "           binding._state.v_split, caps.null_state_index, binding.initial_state_indices.dtype,\n"
        "           parallel._state.reuse_outputs, binding.recurrent_state.dtype)\n",
        f"{MARK}_PREFAXKEY",
        ("B2",),
    ),
    # (22) compile_auxiliary: construir _CommitKernel con state_type.
    (
        B12X_PRE_PARALLEL,
        "        _CommitKernel(**geometry, null_state_index=caps.null_state_index,\n"
        "                      reuse_outputs=parallel._state.reuse_outputs),\n",
        "        _CommitKernel(**geometry, null_state_index=caps.null_state_index,\n"
        "                      reuse_outputs=parallel._state.reuse_outputs,\n"
        f"                      state_type=state_type),  # {MARK}_PREFAXARG (DGX-852 B2)\n",
        f"{MARK}_PREFAXARG",
        ("B2",),
    ),
    # (23) compile_auxiliary: punteros falsos de boundaries/commit -> state_type.
    (
        B12X_PRE_PARALLEL,
        "        (*common, _fake_pointer(BFloat16), _fake_pointer(Int32), _fake_pointer(Float32), Int64(1)),\n"
        "        (*common, _fake_pointer(Float32), _fake_pointer(BFloat16), Int64(1), Int64(1), Int64(1)),\n",
        f"        # {MARK}_PREFAXFAKE (DGX-852 B2): boundaries(source) y commit(state)\n"
        "        # reciben el puntero del estado real en su dtype (base BF16).\n"
        "        (*common, _fake_pointer(BFloat16), _fake_pointer(Int32), _fake_pointer(state_type), Int64(1)),\n"
        "        (*common, _fake_pointer(state_type), _fake_pointer(BFloat16), Int64(1), Int64(1), Int64(1)),\n",
        f"{MARK}_PREFAXFAKE",
        ("B2",),
    ),
    # (24) compile_auxiliary: puntero real de boundaries -> state_type.
    (
        B12X_PRE_PARALLEL,
        "                    _pointer(active.parallel.transfer_flags, Int32),\n"
        "                    _pointer(active.recurrent_state, Float32),\n"
        "                    int(active.recurrent_state.stride(0)), current_cuda_stream())\n",
        "                    _pointer(active.parallel.transfer_flags, Int32),\n"
        f"                    _pointer(active.recurrent_state, state_type),  # {MARK}_PREFAXBND (DGX-852 B2)\n"
        "                    int(active.recurrent_state.stride(0)), current_cuda_stream())\n",
        f"{MARK}_PREFAXBND",
        ("B2",),
    ),
    # (25) compile_auxiliary: puntero real de commit -> state_type.
    (
        B12X_PRE_PARALLEL,
        "        compiled[3](*args(active), _pointer(active.recurrent_state, Float32),\n",
        f"        compiled[3](*args(active), _pointer(active.recurrent_state, state_type),  # {MARK}_PREFAXCOMMIT (DGX-852 B2)\n",
        f"{MARK}_PREFAXCOMMIT",
        ("B2",),
    ),
]


def resolve_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # En la imagen: localizar el paquete SIN ejecutar su __init__ (importar vllm
    # inicializa CUDA en el proceso padre, antes de `vllm serve`). find_spec solo
    # localiza el modulo.
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit(f"{MARK} ABORTA: no puedo localizar vllm instalado; "
                 "pasa --root <dist-packages>")
    return Path(spec.origin).resolve().parent.parent  # .../dist-packages


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="dist-packages con vllm/ y b12x/. Por defecto, el instalado.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Comprueba anclas sin escribir.")
    args = ap.parse_args(argv)

    mix = os.environ.get(MARK, "").strip().upper()
    if mix not in ("", "A", "B", "B2"):
        print(f"{MARK} ABORTA: valor invalido {mix!r} (esperado A, B, B2 o vacio)",
              file=sys.stderr)
        return 2
    if mix == "":
        print(f"{MARK} inerte (vacio): no toco nada")
        return 0

    root = resolve_root(args.root)
    if not (root / "vllm").is_dir() or not (root / "b12x").is_dir():
        print(f"{MARK} ABORTA: {root} no tiene vllm/ y b12x/", file=sys.stderr)
        return 2

    # Agrupar por fichero. FASE 1: leer, aplicar anclas y compilar TODO sin
    # escribir. FASE 2: escribir solo si todo casa y compila. Asi el fail-closed
    # es atomico entre ficheros (B2 toca 5 ficheros: si uno no casa, no se
    # escribe ninguno, no queda la imagen a medias).
    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for rel, old, new, marker, modes in EDITS:
        if mix not in modes:
            continue
        by_file.setdefault(rel, []).append((old, new, marker))

    applied: list[str] = []
    skipped: list[str] = []
    staged: dict[Path, str] = {}
    for rel, edits in by_file.items():
        path = root / rel
        if not path.is_file():
            print(f"{MARK} ABORTA: falta {path}", file=sys.stderr)
            return 3
        text = path.read_text()
        original = text
        for old, new, marker in edits:
            if marker in text:
                skipped.append(f"{rel}#{marker}")
                continue
            n = text.count(old)
            if n != 1:
                print(f"{MARK} ABORTA: ancla no unica o ausente en {rel} "
                      f"(count={n} para {marker}): {old[:72]!r}", file=sys.stderr)
                return 3
            text = text.replace(old, new)
        # Validar sintaxis ANTES de escribir (fail-closed).
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            print(f"{MARK} ABORTA: el fichero parcheado no compila: {rel}: {exc}",
                  file=sys.stderr)
            return 3
        if text != original:
            staged[path] = text
            for _o, _n, marker in edits:
                if marker not in original:
                    applied.append(f"{rel}#{marker}")

    # FASE 2: todo valido -> escribir (salvo dry-run).
    if not args.dry_run:
        for path, text in staged.items():
            path.write_text(text)

    tag = "aplicado" if applied else "ya aplicado"
    print(f"{MARK}={mix} {tag}: " + (", ".join(applied) if applied else "(nada nuevo)"))
    if skipped:
        print(f"{MARK}: ya estaban: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
