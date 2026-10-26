#!/usr/bin/env python3
"""Conecta el lambda POR PETICION con vLLM: scheduler -> worker -> buffer por token.

Companion de `patch_vllm_qwen38next.py` (que ablaciona el modelo). Este archivo NO
toca el modelo: mueve el numero desde `cache_salt` hasta el tensor que lee el grafo.

Cinco ficheros. Anclas verificadas contra el arbol del FORK b12x extraido de
`ghcr.io/ursuciprian/spark-vllm-b12x@sha256:8abb60ee…` (vllm
20260926.dev28+g884b4ff68): 12 de las 14 casan igual que en la nightly de
produccion y 2 estan reubicadas (el constructor de `NewRequestData` y la entrada
del borrador). El runner del fork es el MISMO `v1/worker/gpu/model_runner.py`
(ambas recetas arrancan con VLLM_USE_V2_MODEL_RUNNER=1), asi que el dial por
peticion no cambia de sitio: cambia de ancla en dos puntos.

  SCHED  v1/core/sched/output.py      cache_salt -> NewRequestData.refusal_lambda
  RUNNER v1/worker/gpu/model_runner.py  RefusalState + fill por paso
  WORKER v1/worker/gpu_worker.py       set/get_refusal_lambda (el dial en caliente)
  CACHE  v1/core/kv_cache_utils.py     lambda en la clave de bloque
  SERVE  entrypoints/serve/__init__.py  /refusal_lambda

DIFERENCIA con el port del 27B, y es la unica no cosmetica:
  En 0.27.1 la clave de KV se construia con un guard
      ... or (request.cache_salt is not None)
  y el parche del 27B lo extendia. En este arbol ESE GUARD YA NO EXISTE: vLL lo
  sustituyo por listas de claves (`cache_salt_keys`, kv_cache_utils.py:561). El
  ancla del 27B cuenta 0 aqui. Se reubica al sitio equivalente REAL: anadir el
  lambda a `extra_keys`, que es lo que ahora decide si un bloque es reutilizable.
  Sin esto, dos peticiones con el mismo prompt y distinto lambda COMPARTEN KV: la
  segunda recibe el resultado de la primera, con el lambda equivocado, y ninguna
  señal lo dice.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[qwen38next-runtime] {path}: el ancla aparece {found} veces, "
            f"se esperaban {count}\n--- ancla ---\n{old}\n-------------"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_scheduler(site: Path) -> None:
    p = site / "v1" / "core" / "sched" / "output.py"
    replace(
        p,
        """    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
""",
        """    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
    # rank1-refusal: fuerza de ablacion codificada como cache_salt="refusal:<x>".
    # None = sin sello -> usa el lambda global del servidor.
    refusal_lambda: float | None = None
""",
    )
    replace(
        p,
        """            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            boundary_checkpoint=request.boundary_checkpoint,
""",
        # RE-ANCLADA para el fork (20260926.dev28): el constructor de
        # NewRequestData gano tres kwargs despues de prefill_token_ids
        # (boundary_checkpoint, boundary_checkpoint_blocks,
        # recurrent_instruction_boundary), asi que el ancla vieja, que terminaba
        # en `)`, casaba 0 veces. Se ancla por las dos lineas de arriba y se
        # encaja el lambda ENTRE prefill_token_ids y boundary_checkpoint, que es
        # donde va: `from_request` es el unico constructor del dataclass.
        """            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            refusal_lambda=parse_refusal_lambda(request.cache_salt),
            boundary_checkpoint=request.boundary_checkpoint,
""",
    )
    replace(
        p,
        "from vllm.config.ec_manager_config import EncoderCacheManagerMetadata\n",
        """from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.refusal_projection import parse_request_lambda as parse_refusal_lambda
""",
    )


def patch_model_runner(site: Path) -> None:
    p = site / "v1" / "worker" / "gpu" / "model_runner.py"
    replace(
        p,
        "from vllm.v1.worker.gpu.lora_utils import (\n",
        """from vllm import refusal_projection
from vllm.v1.worker.gpu.refusal_utils import RefusalState
from vllm.v1.worker.gpu.lora_utils import (
""",
    )
    replace(
        p,
        """        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
""",
        """        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        # rank1-refusal: espejo del LoraState. Un slot por peticion.
        self.refusal_state = RefusalState(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
""",
    )
    replace(
        p,
        """        self.lora_state.remove_request(req_id)
        return True
""",
        """        self.lora_state.remove_request(req_id)
        self.refusal_state.remove_request(req_idx)
        return True
""",
    )
    replace(
        p,
        """            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)
""",
        """            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)
            self.refusal_state.add_request(req_index, new_req_data.refusal_lambda)
""",
    )
    # El fill va en el paso real. `num_tokens` es la suma REAL: con verificacion
    # adaptativa sum(num_scheduled_tokens) es una cota y desalinearia los tramos.
    replace(
        p,
        """                self._set_active_loras(*lora_inputs)
        else:
""",
        """                self._set_active_loras(*lora_inputs)
            if refusal_projection.is_enabled():
                _lam = refusal_projection.get_lambda()
                self.refusal_state.fill_target(
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                    _lam,
                    num_tokens=getattr(input_batch, "num_tokens", None),
                )
        else:
""",
    )
    # 03-09-2026: `fill_draft_neutral` estaba AQUI ARRIBA, en el prep del target,
    # seis lineas despues de `fill_target`. Y es un alias de `fill_neutral`, que
    # hace `st.tok.fill_(global)` sobre TODO el buffer: borraba los lambdas por
    # peticion antes de que el target hiciera su forward. Medido en la imagen:
    #     tras fill_target        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    #     tras fill_draft_neutral [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Resultado: el sello por peticion no tenia NINGUN efecto y el lambda global
    # ganaba siempre. El dial global si funcionaba, que es lo que lo disimulaba.
    #
    # La intencion era correcta -- que el drafter no herede el lambda del ultimo
    # lote del target -- pero el sitio no: va en la ENTRADA del drafter.
    # `draft_tokens = self.speculator.propose(` aparece UNA sola vez en el
    # fichero (el dummy run usa `self.speculator.propose(` sin asignar), asi que
    # el ancla no puede colarse en el camino de profiling.
    #
    # RE-ANCLADA para el fork (20260926.dev28): entre el `with use_workspace_lane`
    # y el `propose` se metio el bloque del boundary checkpoint, y el ancla vieja
    # casa 0. Se ancla por las TRES primeras lineas del bloque real:
    # `use_workspace_lane` aparece 3 veces en el fichero (carga del modelo, dummy
    # run del DP-sync y capture), asi que el ancla corta seria ambigua.
    replace(
        p,
        """            with use_workspace_lane(self._draft_workspace_lane):
                if boundary_state is not None:
                    self.speculator.boundary_checkpoint_capture = (
""",
        """            if refusal_projection.is_enabled():
                # El borrador no lleva sello propio: se neutraliza al GLOBAL antes
                # de entrar, y solo aqui. Pegado al propose real, no al dummy run.
                self.refusal_state.fill_draft_neutral(
                    refusal_projection.get_lambda()
                )
            with use_workspace_lane(self._draft_workspace_lane):
                if boundary_state is not None:
                    self.speculator.boundary_checkpoint_capture = (
""",
    )
    replace(
        p,
        """            if not skip_attn_for_dummy_run:
""",
        """            if refusal_projection.is_enabled():
                self.refusal_state.fill_neutral(refusal_projection.get_lambda())
            if not skip_attn_for_dummy_run:
""",
    )


def patch_worker(site: Path) -> None:
    p = site / "v1" / "worker" / "gpu_worker.py"
    replace(
        p,
        """    def sleep(self, level: int = 1) -> None:
""",
        """    def set_refusal_lambda(self, value: float) -> float:
        from vllm.refusal_projection import set_lambda

        applied = set_lambda(float(value))
        logger.info("rank1-refusal: lambda -> %s (rank %s)", applied, self.rank)
        return applied

    def get_refusal_lambda(self) -> float:
        from vllm.refusal_projection import get_lambda

        return get_lambda()

    def sleep(self, level: int = 1) -> None:
""",
    )


def patch_kv_cache(site: Path) -> None:
    """REUBICADO respecto al port del 27B: ver el docstring del modulo."""
    p = site / "v1" / "core" / "kv_cache_utils.py"
    replace(
        p,
        "from vllm.logger import init_logger\n",
        """from vllm.logger import init_logger
from vllm.refusal_projection import lambda_hash_key as refusal_lambda_hash_key
""",
    )
    replace(
        p,
        """    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )
""",
        """    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )
    # rank1-refusal: el lambda entra en la clave de bloque. Sin esto, dos
    # peticiones con el mismo prompt y distinto lambda REUTILIZAN el KV del otro:
    # la segunda recibe el resultado con el lambda equivocado y sin senal.
    # Solo en el primer bloque: el lambda es por peticion, no por bloque.
    _lam_key = refusal_lambda_hash_key()
    refusal_keys: list[str] = (
        [f"refusal_lambda={_lam_key}"]
        if start_token_idx == 0 and _lam_key is not None
        else []
    )

    extra_keys: list[Any] = (
        lora_extra_keys
        + mm_extra_keys
        + cache_salt_keys
        + prompt_embeds_keys
        + refusal_keys
    )
""",
    )


def patch_serve(site: Path) -> None:
    p = site / "entrypoints" / "serve" / "__init__.py"
    replace(
        p,
        """    attach_tokenize_router(app)
""",
        """    attach_tokenize_router(app)

    from vllm.entrypoints.serve.refusal.api_router import (
        attach_router as attach_refusal_router,
    )

    attach_refusal_router(app)
""",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    args = ap.parse_args()
    site = args.site.resolve()

    import shutil

    shutil.copyfile(args.payload / "refusal_utils.py",
                    site / "v1" / "worker" / "gpu" / "refusal_utils.py")

    patch_scheduler(site)
    patch_model_runner(site)
    patch_worker(site)
    patch_kv_cache(site)
    patch_serve(site)
    print("[qwen38next-runtime] lambda por peticion conectado: 5 ficheros")


if __name__ == "__main__":
    main()
