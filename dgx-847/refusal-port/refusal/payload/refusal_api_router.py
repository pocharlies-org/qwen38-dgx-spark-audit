# SPDX-License-Identifier: Apache-2.0
"""Control en caliente de lambda de la proyeccion rank-1 (Qwen3.8-Flash-Next).

  POST /admin/refusal_lambda   {"lambda": 0.0}
  GET  /admin/refusal_lambda

Port del router de `qwen38-27b-rank1-refusal-projection`, con dos cosas de alli
corregidas y una anadida. Las tres por defecto real, no por gusto:

1. `is_enabled()` se comprueba DENTRO de attach_router, no al importar. En el
   27B el `from vllm.refusal_projection import is_enabled` se resuelve al
   importar el modulo, y este vLLM importa `entrypoints/serve/*` antes de
   construir el modelo: el estado todavia no existe, la ruta no se monta nunca.
   Lo mismo aplica a `lambda_hash_key`. Se consulta en la llamada.

2. El 27B usaba `VLLM_REFUSAL_DIRS` (un safetensors con 128 direcciones por
   modulo). Aqui es UNA direccion de 2560 floats: `VLLM_REFUSAL_DIRECTION`. El
   nombre de la variable sale del payload, no de la memoria.

3. collective_rpc, no un setter local: con `--distributed-executor-backend mp`
   y TP=2 a dos nodos, el proceso del frontend NO es el de los workers. Fijar
   lambda aqui sin RPC no llegaria a ningun rank y el dial daria 200 mintiendo.
   Se exige que TODOS los ranks devuelvan el mismo valor; si discrepan, 500.
   Un servidor con lambda distinto por rank produce salida basura sin avisar.

NO EXPONER POR EL INGRESS PUBLICO. vLLM no puede garantizarlo: es una ruta HTTP
mas y quien decide que sale a internet es el ingress. /admin/* no debe salir de
la red del cluster.
"""

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()

# Cota amplia a proposito. 0 = base intacta (proyeccion identidad bit a bit);
# 1.5 = el punto calibrado de ESTE checkpoint (lambda_eff medido 1.4994-1.5008
# sobre 7 tensores, spread 0.09%); >1.5 sobredispara e invierte la componente;
# <0 AMPLIFICA la direccion en vez de eliminarla. La curva de refusal NO es
# monotona: a 2.5 el refusal vuelve a subir, asi que el maximo no es un capricho.
LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0


class RefusalLambdaRequest(BaseModel):
    lambda_: float = Field(..., alias="lambda", ge=LAMBDA_MIN, le=LAMBDA_MAX)

    model_config = {"populate_by_name": True}


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


async def _rpc(raw_request: Request, method: str, args: tuple = ()):
    return await engine_client(raw_request).collective_rpc(method, args=args)


@router.post("/admin/refusal_lambda", dependencies=[Depends(validate_json_request)])
async def set_refusal_lambda(request: RefusalLambdaRequest, raw_request: Request):
    value = float(request.lambda_)
    results = await _rpc(raw_request, "set_refusal_lambda", (value,))

    applied = [r for r in results if r is not None]
    if not applied or any(abs(r - value) > 1e-9 for r in applied):
        logger.error(
            "refusal lambda: los ranks NO coinciden tras fijar %s: %s", value, results
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "los ranks no aplicaron el mismo lambda",
                "requested": value,
                "per_rank": results,
            },
        )

    logger.info("refusal lambda fijado a %s en %d ranks", value, len(applied))
    return JSONResponse(content={"lambda": value, "ranks": len(applied)})


@router.get("/admin/refusal_lambda")
async def get_refusal_lambda(raw_request: Request):
    results = await _rpc(raw_request, "get_refusal_lambda")
    vals = [r for r in results if r is not None]
    consistent = bool(vals) and all(abs(r - vals[0]) <= 1e-9 for r in vals)
    return JSONResponse(
        content={
            "lambda": vals[0] if consistent else None,
            "consistent": consistent,
            "per_rank": results,
        }
    )


def attach_router(app: FastAPI):
    import os

    # Se gatea por VARIABLE DE ENTORNO, no por is_enabled().
    #
    # El estado de refusal_projection nace en init_from_env, y eso ocurre en UN
    # solo sitio: el constructor del modelo (models/qwen3_8_flash_next/model.py
    # en el fork, models/qwen4_exp/nvidia/model.py en la nightly). Con mp + TP=2 el modelo vive en los procesos worker, mientras
    # que este router se monta en el proceso del servidor API, que no construye
    # modelo nunca. Alli is_enabled() es False SIEMPRE, la ruta no se montaba, y
    # /admin/refusal_lambda daba 404 permanente aunque la proyeccion estuviera
    # activa en los workers. La variable si la ven los dos procesos, porque es
    # del contenedor.
    #
    # Es el patron de serve/lora/api_router.py, que gatea por
    # envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING y no por estado en proceso.
    if not (os.environ.get("VLLM_REFUSAL_DIRECTION")
            or os.environ.get("SGLANG_REFUSAL_DIRECTION")):
        # Sin direccion no hay hook que controlar: no se monta. Un endpoint que
        # devuelve 503 en cada llamada es peor que no tenerlo, porque el panel
        # lo ofreceria como dial vivo.
        return
    logger.warning(
        "Proyeccion rank-1 ACTIVA: /admin/refusal_lambda montado. Esta ruta "
        "cambia el comportamiento del modelo en caliente y NO debe salir por "
        "el ingress publico."
    )
    app.include_router(router)
