"""RefusalState: el puente entre el SchedulerOutput y el buffer por token.

Vive en `vllm/v1/worker/gpu/` porque es el unico sitio desde donde se ve a la vez
el `NewRequestData.refusal_lambda` (que pone el scheduler) y el `BatchReqState`
del paso (que dice que tokens van a correr AHORA).

Espejo del `LoraState` del model runner: un slot por peticion, alta/baja con el
ciclo de vida de la peticion, y un fill por paso fuera del grafo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from vllm import refusal_projection as _refusal


class RefusalState:
    """Lambda por peticion, materializado por token en cada paso.

    `lambdas[i]` es None cuando la peticion del slot i NO trajo sello: esa peticion
    usa el global. Distinguir "sin sello" de "sello 0" es obligatorio, porque un
    `refusal:0` explicito pide la base y no el lambda de otra peticion.
    """

    def __init__(self, max_num_reqs: int, max_num_tokens: int, device) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.device = device
        self.lambdas: list[Optional[float]] = [None] * self.max_num_reqs
        # El buffer por token lo posee el payload: es el tensor que el grafo lee.
        # Aqui solo se rellena.
        self._scratch = np.empty(max(1, int(max_num_tokens)), dtype=np.float32)

    def add_request(self, req_idx: int, lam: Optional[float]) -> None:
        i = int(req_idx)
        if i >= self.max_num_reqs:
            # Crecer en caliente bajo CUDA graph no es una opcion: el grajo apunto
            # al buffer. Se avisa y se deja sin sello (usa el global).
            _refusal._warn_once(
                "slot_overflow",
                f"req_idx {i} excede max_num_reqs {self.max_num_reqs}; "
                "esa peticion usara el lambda GLOBAL",
            )
            return
        self.lambdas[i] = None if lam is None else float(lam)
        if lam is not None:
            _refusal.add_request(i, lam)

    def remove_request(self, req_idx: int) -> None:
        i = int(req_idx)
        if 0 <= i < self.max_num_reqs:
            self.lambdas[i] = None
        _refusal.remove_request(i)

    def fill_target(self, idx_mapping_np, num_scheduled_tokens, global_lambda,
                    num_tokens=None) -> None:
        if not _refusal.is_enabled():
            return
        _refusal.fill_target(idx_mapping_np, num_scheduled_tokens, global_lambda,
                             num_tokens=num_tokens)

    def fill_draft_neutral(self, global_lambda) -> None:
        _refusal.fill_draft_neutral(global_lambda)

    def fill_neutral(self, global_lambda) -> None:
        _refusal.fill_neutral(global_lambda)


__all__ = ["RefusalState"]
