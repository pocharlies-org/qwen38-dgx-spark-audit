#!/usr/bin/env python3
"""Cablea la sonda de refusal (FASE B) de punta a punta: forward -> respuesta HTTP.

Corre DESPUES de `patch_vllm_qwen38next.py` y `patch_vllm_runtime.py`: dos de sus
anclas (model.py, model_runner.py) caen en ficheros que esos ya tocaron, y se
eligieron para no solapar con sus inserciones. El modulo es
`payload/refusal_probe.py`; aqui solo se enchufa.

  MODEL   models/qwen3_8_flash_next/model.py   probe(block_input) antes del MLP
  RUNNER  v1/worker/gpu/model_runner.py      step_stats antes del AsyncOutput
  ASYNC   v1/worker/gpu/async_utils.py       tensor CPU -> listas tras copy_event
  MRO     v1/outputs.py                      ModelRunnerOutput.refusal_probe
  SCHED   v1/core/sched/scheduler.py         accumulate por paso, finalize al acabar
  ECO     v1/engine/__init__.py              EngineCoreOutput.refusal_probe (al FINAL:
                                              array_like posicional)
  OPROC   v1/engine/output_processor.py      -> RequestOutput.refusal_probe
  ROUT    outputs.py                         RequestOutput.add conserva el campo
  SERVE   entrypoints/openai/chat_completion/serving.py
            stream:    `provider_specific_fields` en el chunk con finish_reason
            no-stream: clave extra `refusal_probe` en el choice
          Es la forma que LiteLLM ya propaga: top-level `provider_specific_fields`
          del chunk (streaming_handler.copy_model_response_level_...) y claves
          desconocidas del choice -> choice.provider_specific_fields.

Mismas reglas que el resto del mod: cada ancla tiene que aparecer EXACTAMENTE las
veces esperadas o el build muere. Lo que es fail-safe es la sonda en runtime, no
el parcheo: un parche a medias es peor que ninguno.

Con `VLLM_REFUSAL_PROBE` apagado todo lo insertado es una comprobacion de un
booleano: `probe()` sale en la primera linea y `step_stats` no se llama.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[refusal-probe] {path}: el ancla aparece {found} veces, "
            f"se esperaban {count}\n--- ancla ---\n{old}\n-------------"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_model(site: Path) -> None:
    p = site / "models" / "qwen3_8_flash_next" / "model.py"
    replace(
        p,
        "        mlp_out = self.mlp(block_input)\n",
        """        # refusal-probe: lee el tronco multi-stream (N, hc*H) ya combinado con
        # la atencion de esta capa, ANTES del MLP. Solo escribe en un buffer
        # propio; hidden_states y block_input siguen intactos.
        _refusal_probe.probe(
            hidden_states, self.layer_idx, self._rank1_prefix, self.config.hc_count
        )
        mlp_out = self.mlp(block_input)
""",
    )
    # El buffer tiene que existir ANTES de capturar los CUDA graphs, igual que el
    # estado del dial: si naciera dentro de la captura viviria en el pool del
    # grafo. No se fia de que haya un profile run eager antes.
    replace(
        p,
        "        _refusal.init_from_env(int(config.hidden_size))\n",
        """        _refusal.init_from_env(int(config.hidden_size))
        _refusal_probe.init_eager()
""",
    )
    replace(
        p,
        "from vllm import refusal_projection as _refusal\n",
        """from vllm import refusal_projection as _refusal
from vllm import refusal_probe as _refusal_probe
""",
    )


def patch_model_runner(site: Path) -> None:
    p = site / "v1" / "worker" / "gpu" / "model_runner.py"
    replace(
        p,
        "from vllm import refusal_projection\n",
        """from vllm import refusal_projection
from vllm import refusal_probe
""",
    )
    # ANTES del AsyncOutput y a proposito: la copia D2H de step_stats se encola en
    # el main stream, y el AsyncOutput hace copy_stream.wait_stream(main_stream)
    # antes de grabar copy_event. Asi get_output() la encuentra completa sin
    # sincronizar nada nuevo. Y antes de propose(): el drafter no escribe en el
    # buffer (filtro por prefijo), pero leerlo antes no depende de eso.
    # RE-ANCLADA para el fork: el `AsyncOutput(` ya NO va pegado al comentario
    # (entre medias esta el `boundary_state`) y hay DOS construcciones, una por
    # rama del boundary checkpoint. Se ancla por comentario + `boundary_state = (`
    # y la asignacion entra ANTES de las dos, que es lo unico que importa: la copia
    # D2H de step_stats tiene que encolarse en el main stream antes de que
    # AsyncOutput grabe su copy_event. `input_batch.is_prefilling_np` si existe
    # aqui (input_batch.py:89, dataclass del runner v2; el `gpu_input_batch.py`
    # del runner v1 NO lo tiene: buscarlo ahi es un falso negativo).
    replace(
        p,
        """        # Start async output copy here so that it can overlap with speculator proposal.
        boundary_state = (
""",
        """        if refusal_probe.is_active():
            model_runner_output.refusal_probe = refusal_probe.step_stats(
                input_batch.num_tokens,
                input_batch.num_reqs,
                input_batch.query_start_loc,
                input_batch.is_prefilling_np,
            )
        # Start async output copy here so that it can overlap with speculator proposal.
        boundary_state = (
""",
    )


def patch_async_utils(site: Path) -> None:
    p = site / "v1" / "worker" / "gpu" / "async_utils.py"
    replace(
        p,
        "        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict\n",
        """        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        if self.model_runner_output.refusal_probe is not None:
            from vllm import refusal_probe as _refusal_probe

            self.model_runner_output.refusal_probe = _refusal_probe.materialize(
                self.model_runner_output.refusal_probe
            )
""",
    )


def patch_outputs_v1(site: Path) -> None:
    p = site / "v1" / "outputs.py"
    replace(
        p,
        "    sampling_masks: SamplingMaskLists | None = None\n",
        """    sampling_masks: SamplingMaskLists | None = None

    # refusal-probe: {"rows": [[max, sum, n, last_1..K], ...], "prefill": [...]}
    # indexado como req_ids. None con la sonda apagada.
    # `object` y no `Any`: este fichero no importa Any y no merece otra ancla.
    refusal_probe: object = None
""",
    )


def patch_scheduler(site: Path) -> None:
    p = site / "v1" / "core" / "sched" / "scheduler.py"
    replace(
        p,
        "from vllm.logger import init_logger\n",
        """from vllm.logger import init_logger
from vllm import refusal_probe as _refusal_probe
""",
    )
    replace(
        p,
        """            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
""",
        """            req_index = model_runner_output.req_id_to_index[req_id]
            if model_runner_output.refusal_probe is not None:
                _refusal_probe.accumulate(
                    request, model_runner_output.refusal_probe, req_index
                )
            generated_token_ids = (
""",
    )
    replace(
        p,
        """                        num_nans_in_logits=request.num_nans_in_logits,
                    )
""",
        """                        num_nans_in_logits=request.num_nans_in_logits,
                        refusal_probe=(
                            _refusal_probe.finalize(request)
                            if finish_reason is not None
                            else None
                        ),
                    )
""",
    )


def patch_engine_core_output(site: Path) -> None:
    p = site / "v1" / "engine" / "__init__.py"
    replace(
        p,
        """    spec_decode_metrics: RequestSpecDecodeMetrics | None = None

    @property
""",
        """    spec_decode_metrics: RequestSpecDecodeMetrics | None = None

    # refusal-probe: veredicto por peticion, solo en la salida final. Al FINAL de
    # la struct por la misma razon que spec_decode_metrics (array_like).
    refusal_probe: dict[str, Any] | None = None

    @property
""",
    )


def patch_output_processor(site: Path) -> None:
    p = site / "v1" / "engine" / "output_processor.py"
    replace(
        p,
        """                if req_state.streaming_input:
                    request_output.finished = False
""",
        """                if engine_core_output.refusal_probe is not None:
                    request_output.refusal_probe = engine_core_output.refusal_probe
                if req_state.streaming_input:
                    request_output.finished = False
""",
    )


def patch_request_output(site: Path) -> None:
    p = site / "outputs.py"
    # El AsyncLLM fusiona salidas pendientes con add(): el campo viene en la
    # ULTIMA y sin esto se perderia en la fusion.
    replace(
        p,
        "        self.kv_transfer_params = next_output.kv_transfer_params\n",
        """        self.kv_transfer_params = next_output.kv_transfer_params
        _rp = getattr(next_output, "refusal_probe", None)
        if _rp is not None:
            self.refusal_probe = _rp
""",
    )


def patch_serving(site: Path) -> None:
    p = site / "entrypoints" / "openai" / "chat_completion" / "serving.py"
    replace(
        p,
        """                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
""",
        """                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    if choice_data.finish_reason is not None:
                        _rp = getattr(res, "refusal_probe", None)
                        if _rp is not None:
                            chunk.provider_specific_fields = {"refusal_probe": _rp}
""",
    )
    replace(
        p,
        """                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
""",
        """                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
            _rp = getattr(final_res, "refusal_probe", None)
            if _rp is not None:
                choice_data.refusal_probe = _rp
""",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    args = ap.parse_args()
    site = args.site.resolve()
    shutil.copyfile(args.payload / "refusal_probe.py", site / "refusal_probe.py")
    patch_model(site)
    patch_model_runner(site)
    patch_async_utils(site)
    patch_outputs_v1(site)
    patch_scheduler(site)
    patch_engine_core_output(site)
    patch_output_processor(site)
    patch_request_output(site)
    patch_serving(site)
    print("[refusal-probe] sonda cableada: 9 ficheros")


if __name__ == "__main__":
    main()
