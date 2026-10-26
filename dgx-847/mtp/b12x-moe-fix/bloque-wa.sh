# >>> QWEN38_B12X_MOE_WA (DGX-852) >>>
#
# Workaround del IMA del MoE b12x nvfp4 en la banda micro. Evidencia (a-arranque.log y
# b12x-1.3.0): la captura PIECEWISE pasó 9/14 tamanos (M=56..8, todos dynamic tile16 con
# enrutado REAL) y revienta en la 10ª, M=6, con CUDA_LAUNCH_BLOCKING apagado (el IMA
# aparece en el allreduce de la GDN de la capa 0, asincrono). Con topk=10 y el cutover por
# defecto de b12x (B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=64 pares, _impl.py:1853), M<=6 cae en
# la banda MICRO (6*10=60 < 64) y M=8 ya es dynamic (80 >= 64): M=6 es la PRIMERA ejecucion
# del micro nvfp4 con enrutado real de este arranque. El warmup b12x lanzo el micro solo
# con topk_ids ciclicos (expertos 0..9, a-arranque.log:659 "Warmed up 1 b12x MoE kernel
# signature(s)"), nunca con los hasta 60 expertos distintos que produce el router real.
# La geometria del plan/launch es autoconsistente en CPU (test-geometria.py): el fallo esta
# dentro del kernel micro (micro.py), no en el tamano del scratch. El fork de ursu reescribio
# precisamente esta banda (route-plan en host con Triton, nvfp4_phase1/phase2, planned_tile_m)
# y su receta (ops/recipe/env-2x.env) nunca ejecuta el fused-dynamic/micro de 1.3.0 con
# politica B12X_POLICY_MODE=auto: en 1.3.0 no existe ese politica, y aqui no cabe la rueda
# nueva (es la 1.3.0 del wheelhouse).
#
# Lo que hace la llave: mandar la banda micro a dynamic, que es el kernel que YA paso 9
# capturas con enrutado real en este mismo arranque y cuyo compilado es independiente de m
# (la clave de cache de _get_dynamic_kernel excluye m; _impl.py:9411-9433). Cuesta decode
# mas lento en M<=8 (micro era ~2.2x segun el comentario de _resolve_workspace_layout),
# pero arranca. Rollback: QWEN38_B12X_MOE_WA=''.
#
#   QWEN38_B12X_MOE_WA=cutover0  B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=0: select_tp_moe_backend
#                                pide routed_rows < 0, nunca micro -> todo dynamic.
#   QWEN38_B12X_MOE_WA=a16       VLLM_B12X_MOE_FP4_FORCE_A16=1: el oracle nvfp4
#                                (fused_moe/oracle/nvfp4.py:177-179) pasa el MoE nvfp4 a la
#                                familia w4a16 (Trellis), que no toca micro ni dynamic nvfp4.
#                                Solo con QWEN38_B12X_MTP31=1 abajo (usa la misma entrada
#                                ("nvfp4", None) del dict de la glue, ver nota).
#
# Y aparte, un bug LATENTE de la glue que este arranque no llego a pisar (revienta antes,
# en la captura del target): en fused_moe/b12x.py:45 la entrada ("nvfp4", None) -> w4a16
# declara el layout "w13" (up primero) y el fork declara "w31" (gate primero) para el mismo
# esquema; vLLM guarda w13 = [gate; up], o sea gate primero. Es la capa MoE del MTP (grupo
# w4a16 del checkpoint): con "w13" se intercambian gate/up en el swiglu -> borradores
# incorrectos con aceptacion ~0 (o peor) cuando el MTP por fin corra. QWEN38_B12X_MTP31=1
# reescribe esa linea a "w31" (anclado, idempotente, FAIL-CLOSED). Validar ANTES de fiarse:
# micro-repro.py --layer mtp compara b12x contra la referencia dequant con ambos layouts.
if [ -n "${QWEN38_B12X_MOE_WA:-}" ] || [ "${QWEN38_B12X_MTP31:-0}" = "1" ]; then
  if [ "${QWEN38_B12X:-0}" != "1" ] || [ "${QWEN38_B12X_MOE:-0}" != "1" ]; then
    echo "QWEN38_B12X_MOE_WA ABORTA (73): estas llaves solo tienen sentido con QWEN38_B12X=1 y QWEN38_B12X_MOE=1" >&2
    exit 73
  fi
  B12X_GLUE_VLLM=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/model_executor/layers/fused_moe/b12x.py
  case "${QWEN38_B12X_MOE_WA:-}" in
    ""|cutover0|a16) ;;
    *) echo "QWEN38_B12X_MOE_WA ABORTA (73): valor desconocido '${QWEN38_B12X_MOE_WA}' (cutover0|a16)" >&2; exit 73 ;;
  esac
  [ "${QWEN38_B12X_MOE_WA:-}" = "cutover0" ] && export B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=0
  [ "${QWEN38_B12X_MOE_WA:-}" = "a16" ] && export VLLM_B12X_MOE_FP4_FORCE_A16=1

  # FAIL-CLOSED: comprobar el efecto contra la rueda y la glue INSTALADAS, sin GPU (logica
  # pura de enteros; select_tp_moe_backend no toca CUDA).
  QWEN38_B12X_MOE_WA="${QWEN38_B12X_MOE_WA:-}" B12X_GLUE_VLLM="$B12X_GLUE_VLLM" \
  QWEN38_B12X_MTP31="${QWEN38_B12X_MTP31:-0}" python3 - <<'PYEOF' || { echo "QWEN38_B12X_MOE_WA ABORTA (73): ver el mensaje de arriba" >&2; exit 73; }
import os
import pathlib
import sys

wa = os.environ["QWEN38_B12X_MOE_WA"]
if wa == "cutover0":
    from b12x.moe.fused_moe import _impl
    src = pathlib.Path(_impl.__file__).read_text()
    if "B12X_MICRO_DYNAMIC_CUTOVER_PAIRS" not in src:
        print("QWEN38_B12X_MOE_WA ABORTA: la rueda instalada no lee B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", file=sys.stderr)
        sys.exit(1)
    got = {m: _impl.select_tp_moe_backend(num_tokens=m, num_topk=10, quant_mode="nvfp4")
           for m in (1, 2, 3, 4, 6, 8)}
    bad = {m: b for m, b in got.items() if b != "dynamic"}
    if bad:
        print(f"QWEN38_B12X_MOE_WA ABORTA: con cutover=0 la rueda sigue eligiendo micro en {bad}", file=sys.stderr)
        sys.exit(1)
    print(f"QWEN38_B12X_MOE_WA cutover0 ok: M<=8 -> dynamic ({len(got)} tamanos comprobados)")
elif wa == "a16":
    import vllm.envs as envs
    if not envs.VLLM_B12X_MOE_FP4_FORCE_A16:
        print("QWEN38_B12X_MOE_WA ABORTA: VLLM_B12X_MOE_FP4_FORCE_A16 no llego True al proceso", file=sys.stderr)
        sys.exit(1)
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as oracle
    if "VLLM_B12X_MOE_FP4_FORCE_A16" not in pathlib.Path(oracle.__file__).read_text():
        print("QWEN38_B12X_MOE_WA ABORTA: el oracle nvfp4 instalado no mira FORCE_A16", file=sys.stderr)
        sys.exit(1)
    print("QWEN38_B12X_MOE_WA a16 ok: el oracle nvfp4 instalado consume la variable")

if os.environ.get("QWEN38_B12X_MTP31") == "1":
    glue = pathlib.Path(os.environ["B12X_GLUE_VLLM"])
    old = '("nvfp4", None): ("w4a16", "modelopt_nvfp4", "w13"),'
    new = '("nvfp4", None): ("w4a16", "modelopt_nvfp4", "w31"),'
    txt = glue.read_text()
    n_old, n_new = txt.count(old), txt.count(new)
    if n_new == 1 and n_old == 0:
        print("QWEN38_B12X_MTP31 ya aplicado en", glue)
    elif n_old == 1 and n_new == 0:
        glue.write_text(txt.replace(old, new))
        print(f"QWEN38_B12X_MTP31 aplicado: ('nvfp4', None) w13->w31 en {glue}")
    else:
        print(f"QWEN38_B12X_MTP31 ABORTA: ancla ambigua en {glue} (vieja x{n_old}, nueva x{n_new})", file=sys.stderr)
        sys.exit(1)
PYEOF
  echo "QWEN38_B12X_MOE_WA activo rank=${VLLM_NODE_RANK:-?} wa=${QWEN38_B12X_MOE_WA:-none} mtp31=${QWEN38_B12X_MTP31:-0}"
fi
# <<< QWEN38_B12X_MOE_WA <<<
