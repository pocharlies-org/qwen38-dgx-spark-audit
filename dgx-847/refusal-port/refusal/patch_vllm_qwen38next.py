#!/usr/bin/env python3
"""Dial rank-1 de refusal sobre el FORK b12x de ursuciprian (DGX-847).

Es el mismo metodo que `analysis/qwen38-flash-next-abliteration` y que el mod
`ursucipian` de produccion: una direccion r_hat de 2560, `y <- y - lam*r_hat(r_hat.y)`
en los sitios donde los writers editados entregan su salida, lam por peticion via
`cache_salt: "refusal:<x>"`. Lo que cambia aqui es el ANCLAJE, porque cambia la
base: `vllm 20260926.dev28+g884b4ff68` con kernels b12x.

BLANCO REAL (medido sobre el arbol extraido de la imagen, no de memoria):
  vllm/models/qwen3_8_flash_next/model.py   (NO .../nvidia/model.py, NO qwen3_5.py)
  vllm/models/qwen3_8_flash_next/mtp.py

  El registry del fork desvia `Qwen4ExpForConditionalGeneration` y
  `Qwen3_8FlashNextForConditionalGeneration` AMBAS a `vllm.models.qwen3_8_flash_next`
  (registry.py:596-602), y el checkpoint declara `architectures:
  [Qwen4ExpForConditionalGeneration, Qwen3_8FlashNextForConditionalGeneration]`.
  La nightly del mod de produccion llamaba al modulo `qwen4_exp`; el fork lo
  renombr6. `model_executor/models/qwen3_5.py` sigue existiendo (y el modulo nuevo
  IMPORTA de el `Qwen3_5Model`, para los WeightMappers) pero no se instancia:
  anclar ahi produce una imagen sana que no abla nada. El gate de identidad de
  abajo lo impide.

SITIOS DEL TARGET -- 6, y son los mismos 6 del port de produccion:
  T0 import            model.py
  T1 init + declaracion del sitio de embedding (antes de capturar grafos)
  T2 embed_tokens      DENTRO del `else`: la torre visual entra por inputs_embeds
                       y el checkpoint la deja byte-identica (medido: embed_tokens
                       BF16 identico al de RadixArk, que a su vez es byte-identico
                       al base segun su audit_unchanged_report)
  T3 ple               suma DIRECTA al tronco hc, sin combine -> sitio propio.
                       El fork anade `prefetched=` a la llamada: ancla reescrita.
  T4 attn_out          antes del recorte por `output_indices` (ver abajo)
  T5 mlp_out           cubre los 512 expertos fusionados + shared de una vez
  T6 guard fail-closed

SITIOS DEL DRAFTER MTP -- 3, y aqui hay un cambio de sitio REAL, no cosmetico:
  M0 import
  M1 embed_input_ids           igual que en la nightly
  M2 prologo fc_embedding+fc_hidden: EN LA NIGHTLY se proyectaban los dos writers
     por separado en el prologo Python. En el fork ese prologo es un KERNEL FUSIONADO
     (`_prepare_feedback` -> b12x `mtp_feedback`, que lee `fc_embedding.weight` y
     `fc_hidden.weight` directamente: mtp.py:568-569 y el comentario de mtp.py:782).
     No hay dos lineales que interceptar. Se proyecta SU SALIDA, que es la suma de
     los dos en espacio hc: por linealidad es la misma proyeccion, y sale un sitio
     en vez de dos.

POR QUE T4 VA ANTES DEL RECORTE (descubierto aqui, no existia en la nightly):
  El fork indexa `hidden_states`/`attn_out`/`injection` por `output_indices` entre
  la atencion y el MLP (compactacion de salidas del drafter,
  VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT, default 1). El buffer de lambda esta
  indexado por POSICION DE TOKEN EN EL LOTE, asi que proyectar despues del recorte
  usaria el lambda de otra fila. Con `--max-num-seqs 16` (las dos recetas) la
  compactacion esta apagada -- solo se activa con max_num_seqs==1 -- pero el sitio
  se pone donde es correcto en los dos casos.

TENSOR PARALLEL: sin cambios respecto a produccion, y la razon es la misma capa
  por capa, no la convencion: `attn_out` y `mlp_out` salen all-reduced
  (RowParallel reduce_results=True / MoE all-reduced), `embed_tokens` es
  VocabParallel (particiona VOCAB, no hidden) y el PLE emite al tronco hc ya
  completo. Ningun sitio pasa `sharded=True`. En el TP=2 del fork esto es mas
  importante todavia: su all-reduce es RoCE propio (VLLM_ENABLE_ROCE_ALLREDUCE=1),
  y un gather anadido a ciegas entraria en el camino que ellos miden.

FAIL-CLOSED, con una rama nueva por el AOT compile:
  El fork arranca con VLLM_USE_AOT_COMPILE=1 y VLLM_USE_MEGA_AOT_ARTIFACT=1, y la
  receta siembra la caché desde /opt/b12x-seed. Si el artefacto se carga de disco,
  Dynamo no traza en este proceso y `_consumed` queda vacio: el guard de produccion
  mataria un servidor SANO en el segundo arranque. Por eso el guard elige su
  fuente: `_consumed` (ejecucion) cuando el grafo se compila en el arranque,
  `_declared` (construccion) cuando viene del cache. La prueba de que el hook esta
  dentro del grafo cargado es entonces el checksum de fuente que vLLM ya aplica al
  cargar (`_verify_source_unchanged`: si el .py cambio, descarta y recompila) mas
  la sonda de refusal y el diff greedy lambda=0.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PKG = "qwen3_8_flash_next"          # el fork renombra el modulo de la nightly
PAYLOAD = "refusal_projection.py"


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[b12x-rank1] {path}: el ancla aparece {found} veces, "
            f"se esperaban {count}\n--- ancla ---\n{old}\n-------------"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_model(site: Path) -> None:
    p = site / "models" / PKG / "model.py"

    # T0 ------------------------------------------------------------------ import
    replace(
        p,
        "from vllm.distributed import get_pp_group\n",
        "from vllm.distributed import get_pp_group\n"
        "from vllm import refusal_projection as _refusal\n",
    )

    # T1 ------------------------------------------- init del estado + declaracion
    replace(
        p,
        """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
""",
        """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # rank1-refusal: direccion + tensor lambda + buffer por token. TIENE que
        # existir antes de que se capturen los CUDA graphs: si la primera pasada
        # fuera la captura, el grafo hornea la rama "desactivada" y el dial no
        # mueve nada en silencio. Perezoso no vale.
        _refusal.init_from_env(int(config.hidden_size))
        self._rank1_prefix = prefix
        _refusal.declare(f"{self._rank1_prefix}.embed_tokens")
""",
    )

    # T1b ------------------------------------- prefijo de la capa, al construir
    # En la nightly el forward nombraba sus sitios con `prefix`, que no vive en su
    # ambito: NameError en el primer forward (03-09-2026, cuatro sitios afectados y
    # solo caia el primero que se ejecutaba). El forward escribe la clave como
    # `f"{self._rank1_prefix}.<sitio>.<tipo>"`, con la MISMA expresion del
    # declare: `check_guard_counts.py` lee el patron de esa f-string para casar
    # las cuentas del guard, y un atributo (`self._rk_attn`) le deja de ver un
    # nombre suelto. Dos copias de la expresion, si, pero identicas y a la vista.
    replace(
        p,
        """        self.layer_idx = extract_layer_index(prefix)
""",
        """        self.layer_idx = extract_layer_index(prefix)
        # rank1-refusal: prefijo de la capa, para las claves y para la sonda.
        self._rank1_prefix = prefix
""",
    )

    # T1c -------------------------------------- declaracion de los sitios de capa
    replace(
        p,
        """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            workspace,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )
""",
        """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            workspace,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )
        # rank1-refusal: los tres sitios de la capa, declarados aqui. El tipo de
        # self.mlp se conoce ya (MoE o denso), igual que self.layer_type y que si
        # hay PLE: la declaracion es exacta y el guard de construccion puede valer
        # con el grafo AOT cargado de disco, donde no hay trazado que contar.
        # Las expresiones son las mismas que usan T3/T4/T5 en el forward.
        _refusal.declare(f"{self._rank1_prefix}.attn_out.{self.layer_type}")
        _refusal.declare(f"{self._rank1_prefix}.mlp_out.{type(self.mlp).__name__}")
        if self.ple is not None:
            _refusal.declare(f"{self._rank1_prefix}.ple")
""",
    )

    # T2 ---------------------------------------------------------- embed_tokens
    replace(
        p,
        """                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
""",
        """                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
                # rank1-refusal: `embed_tokens` es writer editado (`kind=in`, se le
                # resta la componente a cada FILA) y la salida de un token ES su
                # fila. Va ANTES del repeat: proyectar y luego replicar a 4 streams
                # sale igual que proyectar 4 veces, y cuesta una.
                hidden_states = _refusal.project(
                    hidden_states, key=f"{self._rank1_prefix}.embed_tokens"
                )
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
""",
    )

    # T3 --------------------------------------------------------------------- ple
    # El fork anade `prefetched=ple_prefetched` a la llamada (overlap de PLE).
    replace(
        p,
        """            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                prefetched=ple_prefetched,
            )
""",
        """            # rank1-refusal: `ple.value_proj` escribe al tronco SIN pasar por
            # ningun combine, asi que necesita sitio propio. Sale en espacio hc
            # aplanado (N, hc*H) -> project_hc. `ple.key_proj` solo forma un gate
            # escalar: NO se toca, corromperia el gate.
            hidden_states = hidden_states + _refusal.project_hc(
                self.ple(
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    prefetched=ple_prefetched,
                ),
                self.config.hc_count,
                key=f"{self._rank1_prefix}.ple",
            )
""",
    )

    # T4 ------------------------------------------------------------- attn_out
    # ANTES del recorte por output_indices: el buffer de lambda va por posicion de
    # token del lote. Ver docstring del modulo.
    replace(
        p,
        """        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        else:
            attn_out = self.self_attn(hidden_states=block_input, positions=positions)
        if output_indices is not None:
""",
        """        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        else:
            attn_out = self.self_attn(hidden_states=block_input, positions=positions)
        # rank1-refusal: proyectar la salida de bloque ANTES de inyectarla
        # equivale a ortogonalizar TODOS los writers que la suman (linealidad).
        # Aqui cubre o_proj / linear_attn.out_proj de una sola vez.
        attn_out = _refusal.project(
            attn_out, key=f"{self._rank1_prefix}.attn_out.{self.layer_type}"
        )
        if output_indices is not None:
""",
    )

    # T5 -------------------------------------------------------------- mlp_out
    replace(
        p,
        """        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
""",
        """        mlp_out = self.mlp(block_input)
        # rank1-refusal: idem, y aqui cubre de una vez los 512 expertos
        # fusionados + el shared expert de la capa.
        mlp_out = _refusal.project(
            mlp_out, key=f"{self._rank1_prefix}.mlp_out.{type(self.mlp).__name__}"
        )
        return hidden_states, mlp_out, injection
""",
    )

    # T6 ------------------------------------------------------------ el guard
    replace(
        p,
        """        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count
        )
""",
        """        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        if _refusal.is_enabled() and (
            self.start_layer == 0 and self.end_layer == config.num_hidden_layers
        ):
            # FAIL-CLOSED, y por PREFIJO + SUFIJO: el drafter MTP vive en el mismo
            # proceso y sus sitios (`<prefix>.mtp.layers.48.attn_out...`) se
            # sumarían a un conteo global. Contar por `startswith` sin sufijo ya
            # mato un servidor sano una vez (97 frente a 96, 03-09-2026).
            #
            # El guard corre en un forward_hook, no en el __init__: los sitios de
            # EJECUCION se registran dentro de project(), que no ha corrido todavia
            # cuando se construye el modelo. Llamalo aqui y mueres siempre (03-09,
            # 16 reinicios del residente).
            _n = config.num_hidden_layers
            _ple = len({abs(i) for i in getattr(config, "ple_layer_ids", [])
                        if 1 <= abs(i) <= _n})

            def _rank1_verify(_mod, _inp, _out, _n=_n, _ple=_ple, _prefix=prefix):
                if getattr(_mod, "_rank1_verified", False):
                    return
                _mod._rank1_verified = True
                # Dos ramas explicitas y con sufijos LITERALES: las llamadas
                # `verify_consumed` de aqui son las que lee `check_guard_counts.py`
                # para casar las cuentas esperadas con los sitios que registra el
                # propio parche. Un bucle `for suffix in (...)` con `suffix=suffix`
                # le deja de ver patron y el gate se queda ciego (y este gate existe
                # para no escribir los numeros a mano).
                if getattr(
                    _mod, "was_aot_compile_fn_loaded_from_disk", False
                ):
                    # El grafo viene serializado del artefacto AOT: `project()` no
                    # corre como Python en ESTE proceso, asi que `_consumed` esta
                    # vacio POR DISENO. Vale el guard de construccion (`declared`,
                    # lleno en el __init__, siempre eager); la ejecucion la acreditan
                    # el checksum de fuente que vLLM verifica al cargar y la sonda.
                    _refusal.verify_declared(
                        f"{_prefix}.layers", _n, suffix="attn_out"
                    )
                    _refusal.verify_declared(
                        f"{_prefix}.layers", _n, suffix="mlp_out"
                    )
                    _refusal.verify_declared(
                        f"{_prefix}.layers", _ple, suffix="ple"
                    )
                else:
                    _refusal.verify_consumed(
                        f"{_prefix}.layers", _n, suffix="attn_out"
                    )
                    _refusal.verify_consumed(
                        f"{_prefix}.layers", _n, suffix="mlp_out"
                    )
                    _refusal.verify_consumed(
                        f"{_prefix}.layers", _ple, suffix="ple"
                    )
                _refusal.verify_keys(f"{_prefix}.embed_tokens", source="declared")
                # model.py NO tiene `logger` propio (mtp.py si): si se nombra aqui,
                # NameError dentro del hook y el arranque muere en el primer
                # forward. verify_declared/verify_consumed ya registran su cuenta.

            self.register_forward_hook(_rank1_verify)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count
        )
""",
    )


def patch_mtp(site: Path) -> None:
    p = site / "models" / PKG / "mtp.py"

    # M0 ------------------------------------------------------------------ import
    replace(
        p,
        "from vllm.distributed import get_pp_group\n",
        "from vllm.distributed import get_pp_group\n"
        "from vllm import refusal_projection as _refusal\n",
    )

    # M1 ------------------------------------------------- embed_input_ids del draft
    replace(
        p,
        """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
""",
        """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # rank1-refusal: el drafter tiene su propia tabla (cargada del
        # `embed_tokens` del target via _remap_mtp_weight_name) y NO pasa por el
        # sitio T2 del target.
        return _refusal.project(
            self.embed_tokens(input_ids), key=self._rk_embed
        )
""",
    )

    # M1b -------------------------------------- declaracion + guard del drafter
    replace(
        p,
        """        self.prefix = prefix
""",
        """        self.prefix = prefix
        # rank1-refusal: los dos sitios propios del prologo del drafter, declarados
        # aqui (el forward del prologo es un kernel fusionado, ver M2).
        self._rk_embed = "mtp.embed_tokens"
        self._rk_feedback = "mtp.feedback"
        _refusal.declare(self._rk_embed)
        _refusal.declare(self._rk_feedback)

        if _refusal.is_enabled():
            # La capa decoder del drafter es la MISMA clase que la del target, asi
            # que sus sitios attn_out/mlp_out los declara y los proyecta el parche
            # del target (prefijo `<...>.mtp.layers.<48>`). Aqui solo se acreditan
            # los sitios del prologo y la cuenta de la capa, con el prefijo del
            # drafter para no mezclarlos con los del target.
            _lp = self.layers[0]._rank1_prefix

            def _rank1_verify_draft(_mod, _inp, _out, _lp=_lp):
                if getattr(_mod, "_rank1_verified", False):
                    return
                _mod._rank1_verified = True
                from_aot_cache = bool(getattr(
                    _mod, "was_aot_compile_fn_loaded_from_disk", False))
                guard = (_refusal.verify_declared if from_aot_cache
                         else _refusal.verify_consumed)
                guard(_lp, 1, suffix="attn_out")
                guard(_lp, 1, suffix="mlp_out")
                _refusal.verify_keys("mtp.embed_tokens", "mtp.feedback",
                                     source="declared")

            self.register_forward_hook(_rank1_verify_draft)
""",
    )

    # M2 --------------------------------- prologo fc_embedding + fc_hidden (kernel)
    # El prologue del drafter ya NO son dos lineales de Python: es el kernel b12x
    # `mtp_feedback`, que recibe `fc_embedding.weight` y `fc_hidden.weight` y los
    # aplica dentro (mtp.py:568-569). Se proyecta la SALIDA del kernel, que es la
    # suma de los dos en espacio hc: por linealidad es la misma proyeccion.
    replace(
        p,
        """        return self._feedback_output[: token_embedding.shape[0]].flatten(-2)
""",
        """        # rank1-refusal: `fc_embedding` y `fc_hidden` son los writers del prologo
        # (los analogos MTP de embed_tokens; medidos: lam_eff 1.5008/1.4994,
        # |cos| 0.999992/0.999998 con la direccion del target). Se abla a
        # proposito: sin esto, con lambda>0 la acceptance del borrador se hunde.
        # Va DESPUES del flatten: el contrato de project_hc es ultimo eje hc*H.
        return _refusal.project_hc(
            self._feedback_output[: token_embedding.shape[0]].flatten(-2),
            self.hc_count,
            key=self._rk_feedback,
        )
""",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True,
                    help="directorio del paquete vllm")
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--direction", type=Path, required=True)
    args = ap.parse_args()

    site = args.site.resolve()

    # GATE DE IDENTIDAD. No basta con que el fichero exista: el fallo historico de
    # esta linea de trabajo fue anclar en un modulo que existe y no se instancia.
    model_py = site / "models" / PKG / "model.py"
    if not model_py.exists():
        raise SystemExit(
            f"[b12x-rank1] no existe {model_py}: esta base no es el fork b12x "
            f"(se espera vllm/models/{PKG}/model.py)"
        )
    mtp_py = site / "models" / PKG / "mtp.py"
    if not mtp_py.exists():
        raise SystemExit(f"[b12x-rank1] no existe {mtp_py}")
    reg = site / "model_executor" / "models" / "registry.py"
    if not reg.exists():
        raise SystemExit(f"[b12x-rank1] no existe {reg}")
    regtext = reg.read_text(encoding="utf-8")
    # El registry del fork declara el modulo con su ruta larga
    # (`"vllm.models.qwen3_8_flash_next"`, registry.py:115/597/697), no el nombre
    # suelto: buscar `"{PKG}"` a secas mataba el arranque de un servidor sano.
    for needed in (f'"vllm.models.{PKG}"', "Qwen4ExpForConditionalGeneration"):
        if needed not in regtext:
            raise SystemExit(
                f"[b12x-rank1] el registry no declara {needed}: el modelo servido "
                "no es el que este parche ablaciona"
            )
    # b12x tiene que estar: sin su loader y sus kernels esto no es la imagen del
    # fork, y el anclaje M2 (prologo fusionado en kernel) no se sostiene.
    if not (site / "utils" / "b12x.py").is_file():
        raise SystemExit(
            "[b12x-rank1] no existe vllm/utils/b12x.py: esta base no lleva los "
            "kernels b12x y el anclaje del prologo MTP no aplica"
        )

    shutil.copyfile(args.payload / PAYLOAD, site / PAYLOAD)
    shutil.copyfile(args.direction, site / "refusal_direction_flashnext.npy")

    patch_model(site)
    patch_mtp(site)
    print(f"[b12x-rank1] anclaje aplicado sobre {PKG}: 4 sitios en el target "
          "(embed, ple, attn_out, mlp_out) + 2 en el drafter (embed, feedback)")


if __name__ == "__main__":
    main()
