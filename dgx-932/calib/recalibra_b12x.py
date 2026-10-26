#!/usr/bin/env python3
"""DGX-932: ajusta fuera del servidor las consultas sequence.gdn_prefill que volcó el TP=2 (VLLM_B12X_DUMP_QUERIES).

Cada consulta única del volcado (la clave de b12x no lleva las ranuras de estado) se reconstruye con la API de b12x y se
exige que su consulta codificada y su invocación sean idénticas a las volcadas. Se ajusta con una PreparationSession con
el espacio de nombres del servidor, leído del fichero de selecciones de producción ya copiado en la caché de destino:
las selecciones nuevas se añaden a ese mismo fichero y el servidor las encuentra al arrancar. La prueba es la de vLLM
(`_b12x_gdn_prefill_call(benchmark=True)`): una secuencia de `capacidad` tokens sobre la ranura 1.

--comprobar no ajusta: calcula la clave de cada consulta como `_choice_key` y la busca en caché, y también la de su
gemela en FP32 (las semillas de ursu). Una gemela encontrada prueba que la clave casa antes de horas de ajuste. Corre en la imagen del TP=2 con el parche B2 ya aplicado (QWEN_GDN_MIX=B2 patch_gdn_mix.py).
"""
import argparse
import json
import logging
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("volcado", help="jsonl de VLLM_B12X_DUMP_QUERIES")
    ap.add_argument("--seleccion", required=True, help="fichero <identidad>.json de producción, en la caché de destino")
    ap.add_argument("--ranuras", type=int, default=64, help="ranuras de estado de la prueba (no entran en la clave)")
    ap.add_argument("--compiladores", type=int, default=16)
    ap.add_argument("--comprobar", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

    import torch
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
    from b12x.preparation._cache import digest
    from b12x.sequence import gdn_prefill
    from b12x.sequence.gdn_prefill import _impl, _tuning
    from b12x.sequence._shared.delta_prefill.preparation import _caps_from_query

    contract = _tuning.TUNING
    filas = [json.loads(line) for line in open(args.volcado) if line.strip()]
    filas = [f for f in filas if f["component"] == contract.component_id]
    assert filas, f"el volcado no trae consultas de {contract.component_id}"
    unicas = {}
    for f in filas:
        assert f["query_schema"] == contract.query_schema_version, ("esquema de consulta distinto", f["query_schema"])
        unicas.setdefault(json.dumps([f["query"], f["invocation"]], sort_keys=True), f)
    print(f"{len(filas)} declaraciones de {contract.component_id}, {len(unicas)} consultas únicas", flush=True)

    device = torch.device("cuda", 0)
    seleccion = json.load(open(args.seleccion))
    session = PreparationSession(device=device, autotune=True, namespace=seleccion["identity"]["namespace"],
                                 compile_workers=args.compiladores)
    cache = session._selection_cache()
    assert str(cache.path) == args.seleccion, f"la identidad no casa: b12x escribiría en {cache.path}"

    def plan_de(f, state_dtype=None):
        query = dict(f["query"], max_state_slots=args.ranuras)
        if state_dtype:
            query["state_dtype"] = state_dtype
        caps = _caps_from_query(_impl, _tuning.GdnPrefillQuery(**query), device.index, is_gdn=True)
        plan = gdn_prefill.plan(caps, invocation=FrozenMapping(f["invocation"]))
        if state_dtype is None:
            assert json.loads(json.dumps(contract.encode_query(plan.query))) == f["query"], ("consulta distinta", f["query"])
            assert plan.invocation.to_dict() == f["invocation"], ("invocación distinta", f["invocation"])
        return caps, plan

    def clave(plan):  # = PreparationJob._choice_key sin dependencias ni pin (el GDN no tiene)
        return digest({
            "component": contract.component_id, "query_schema": contract.query_schema_version,
            "config_schema": contract.config_schema_version, "semantic_version": contract.semantic_version,
            "candidate_contract_version": contract.candidate_contract_version,
            "query": FrozenMapping(contract.encode_query(plan.query)).to_dict(),
            "invocation": plan.invocation.to_dict(), "pin": None, "dependencies": [],
        })

    planes = [(f, *plan_de(f)) for f in unicas.values()]
    gemelas = sum(clave(plan_de(f, "float32")[1]) in cache.records for f in unicas.values())
    en_cache = sum(clave(plan) in cache.records for _, _, plan in planes)
    print(f"en caché: {en_cache}/{len(planes)} consultas; gemelas FP32 (semillas de ursu): {gemelas}/{len(planes)}", flush=True)
    if args.comprobar:
        # Una clave es un sha256: con que una gemela FP32 esté en caché, la fórmula reproduce la del servidor.
        if not gemelas:
            sys.exit("COMPROBACION FALLIDA: ninguna gemela FP32 en caché; la herramienta no reproduce la clave del servidor")
        print("COMPROBADO", flush=True)
        return 0

    def valores(tensor):  # = GatedDeltaNetAttention._benchmark_values
        valores_ = torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device).reshape_as(tensor)
        tensor.copy_(valores_.div_(max(valores_.numel(), 1)))

    def llamada(caps, f):
        kh, vh, n, capacidad = caps.key_heads, caps.value_heads, caps.max_seqs, caps.max_tokens
        estado = torch.zeros((caps.max_state_slots, vh, 128, 128), dtype=caps.state_dtype, device=device)
        A_log = torch.full((vh,), -1.0, dtype=getattr(torch, f["invocation"]["a_log_dtype"]), device=device)
        dt_bias = torch.zeros((vh,), dtype=getattr(torch, f["invocation"]["dt_bias_dtype"]), device=device)

        def call(state):
            spec, = tuple(state.layout.scratch_specs())
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            mixed_qkv = torch.empty((capacidad, (2 * kh + vh) * 128), dtype=torch.bfloat16, device=device)
            a = torch.empty((capacidad, vh), dtype=torch.bfloat16, device=device)
            b = torch.empty_like(a)
            output = torch.empty((capacidad, vh, 128), dtype=torch.bfloat16, device=device)
            cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
            ini, fin, chk = (torch.ones(n, dtype=torch.int32, device=device) for _ in range(3))
            off = torch.zeros(n, dtype=torch.int32, device=device)
            nseq, ntok = torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device)
            guardado = estado[1:2].clone()
            q, k, v = mixed_qkv.split((kh * 128, kh * 128, vh * 128), dim=-1)
            q, k, v = q.view(capacidad, kh, 128), k.view(capacidad, kh, 128), v.view(capacidad, vh, 128)

            def produce():
                for t in (q, k, v, a, b):
                    valores(t)
                cu.zero_(); cu[1:].fill_(capacidad)
                ini.fill_(1); fin.fill_(1); chk.fill_(1); off.zero_(); nseq.fill_(1); ntok.fill_(capacidad)

            def reset():
                estado[1:2].copy_(guardado)

            binding = state.bind(
                scratch=scratch, q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, recurrent_state=estado,
                cu_seqlens=cu, initial_state_indices=ini, final_state_indices=fin, checkpoint_state_indices=chk,
                checkpoint_offsets=off, num_seqs=nseq, num_tokens=ntok, output=output,
            )
            return PreparedCall(run=lambda: state.run(binding, max_live_tokens=capacidad, max_live_seqs=1),
                                produce=produce, reset=reset, restore=reset,
                                owners=(scratch, mixed_qkv, a, b, output, cu, ini, fin, chk, off, nseq, ntok))
        return call

    pendientes = [(f, caps, plan) for f, caps, plan in planes if clave(plan) not in cache.records]
    requests = [plan.request(name=f"gdn_prefill.m{caps.max_tokens}.{i}", prepare_call=llamada(caps, f),
                             benchmark_call=llamada(caps, f)) for i, (f, caps, plan) in enumerate(pendientes)]
    print(f"ajustando {len(requests)} consultas con {args.compiladores} compiladores", flush=True)
    t0, ultimo = time.time(), [0.0]

    def progreso(p):
        if time.time() - ultimo[0] >= 60 or p.done:
            ultimo[0] = time.time()
            print(f"PROGRESO {int(time.time() - t0) // 60} min: {p.completed_requests}/{p.total_requests} listas, "
                  f"{p.request_name} candidatos {p.candidates_prepared}/{p.candidate_count}, medidos {p.measured_candidates}, "
                  f"compilaciones {p.compilations} ({p.active_compilations} activas)", flush=True)

    with session:
        session.prepare(requests, progress=progreso)
    nuevo = json.load(open(args.seleccion))["records"]
    hechas = [clave(plan) in nuevo for _, _, plan in pendientes]
    for (f, caps, plan), ok in zip(pendientes, hechas):
        r = nuevo.get(clave(plan), {})
        print(f"m{caps.max_tokens}: {'guardada' if ok else 'SIN GUARDAR'} {r.get('config')} {r.get('coverage')}", flush=True)
    print(f"{'HECHO' if all(hechas) else 'INCOMPLETO'}: {sum(hechas)}/{len(hechas)} selecciones nuevas en {args.seleccion} "
          f"en {int(time.time() - t0) // 60} min", flush=True)
    return 0 if all(hechas) else 1


if __name__ == "__main__":
    sys.exit(main())
