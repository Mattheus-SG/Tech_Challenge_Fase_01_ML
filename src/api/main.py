"""API de inferência de churn (FastAPI).

Rodar localmente: uv run uvicorn src.api.main:app --reload
Documentação interativa: http://localhost:8000/docs
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.schemas import Cliente, Predicao, Saude
from src.config import LIMIAR_DECISAO, MODEL_PATH, caminho_metadata
from src.predict import carregar_modelo, prever

VERSAO_API = "0.1.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModeloServido:
    """Pipeline treinado e a identificação que vai nas respostas."""

    pipeline: Any
    nome: str
    versao: str


@lru_cache
def carregar_modelo_servido() -> ModeloServido:
    """Carrega o modelo e os metadados uma única vez; as chamadas seguintes usam o cache."""
    pipeline = carregar_modelo(MODEL_PATH)
    # Um cliente por requisição: paralelizar as árvores só adiciona overhead (19 ms -> 9,5 ms).
    # A concorrência vem dos workers do uvicorn, não de threads dentro de cada predição.
    pipeline.set_params(classifier__n_jobs=1)
    metadata = json.loads(caminho_metadata(MODEL_PATH).read_text(encoding="utf-8"))
    return ModeloServido(pipeline, nome=metadata["modelo"], versao=metadata["criado_em"])


def obter_modelo() -> ModeloServido:
    """Dependência do /predict: devolve o modelo ou responde 503 se ele não puder ser carregado."""
    try:
        return carregar_modelo_servido()
    except Exception:
        logger.exception("Falha ao carregar o modelo de %s", MODEL_PATH)
        raise HTTPException(status_code=503, detail="Modelo indisponível no momento") from None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo na subida da API, para a primeira requisição não pagar esse custo."""
    try:
        modelo = obter_modelo()
        logger.info("Modelo carregado: %s (versão %s)", modelo.nome, modelo.versao)
    except HTTPException:
        logger.warning("API no ar sem modelo: /health responde degraded e /predict, 503")
    yield


app = FastAPI(
    title="API de Propensão de Churn",
    description=(
        "Tech Challenge Fase 1 (12MLET, grupo 28): recebe os dados de um cliente de "
        "telecom e devolve a propensão de cancelamento (churn)."
    ),
    version=VERSAO_API,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def erro_de_validacao(request: Request, erro: RequestValidationError) -> JSONResponse:
    """Responde 422 sem repetir o valor recebido: é dado do cliente e pode não caber em JSON."""
    detalhes = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in erro.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(detalhes)})


@app.middleware("http")
async def medir_latencia(request: Request, call_next):
    """Registra método, rota, status e latência de cada requisição (sem dados do cliente)."""
    inicio = time.perf_counter()
    resposta = await call_next(request)
    duracao = time.perf_counter() - inicio
    resposta.headers["X-Process-Time"] = f"{duracao:.4f}"
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        resposta.status_code,
        duracao * 1000,
    )
    return resposta


@app.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    """Abre a documentação interativa."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=Saude,
    responses={503: {"model": Saude, "description": "API no ar, mas sem modelo carregado"}},
)
def health(response: Response) -> Saude:
    """Verifica se a API está no ar e se o modelo está carregado."""
    try:
        modelo = carregar_modelo_servido()
    except Exception:
        # 503 para que orquestradores (Docker, Kubernetes) tirem a instância do ar
        response.status_code = 503
        return Saude(
            status="degraded",
            modelo_carregado=False,
            modelo=None,
            versao_modelo=None,
            versao_api=VERSAO_API,
        )
    return Saude(
        status="healthy",
        modelo_carregado=True,
        modelo=modelo.nome,
        versao_modelo=modelo.versao,
        versao_api=VERSAO_API,
    )


@app.post(
    "/predict",
    response_model=Predicao,
    responses={
        503: {"description": "Modelo indisponível"},
        500: {"description": "Erro interno na inferência"},
    },
)
def predict(
    cliente: Cliente,
    modelo: Annotated[ModeloServido, Depends(obter_modelo)],
) -> Predicao:
    """Recebe os dados de um cliente e devolve a propensão de churn."""
    try:
        [resultado] = prever(modelo.pipeline, cliente.model_dump(), limiar=LIMIAR_DECISAO)
    except Exception:
        # Detalhes só no log do servidor; o cliente recebe uma mensagem genérica
        logger.exception("Erro inesperado na inferência")
        raise HTTPException(status_code=500, detail="Erro interno ao calcular a predição") from None
    return Predicao(**resultado, limiar=LIMIAR_DECISAO, versao_modelo=modelo.versao)
