"""Inferência: carrega o modelo treinado e calcula a propensão de churn de clientes.

Independente de HTTP: usado pela API (src/api) e reutilizável em scripts e notebooks.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURES, LIMIAR_DECISAO, MODEL_PATH

Cliente = Mapping[str, Any]


def carregar_modelo(caminho: Path = MODEL_PATH) -> Pipeline:
    """Carrega o pipeline treinado (pré-processamento + modelo) salvo em .joblib."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}")
    return joblib.load(caminho)


def montar_entrada(clientes: Cliente | Sequence[Cliente]) -> pd.DataFrame:
    """Converte um cliente (dict) ou uma lista de clientes no DataFrame que o pipeline espera."""
    registros = [clientes] if isinstance(clientes, Mapping) else list(clientes)
    if not registros:
        raise ValueError("Nenhum cliente informado")

    df = pd.DataFrame(registros)
    ausentes = [c for c in FEATURES if c not in df.columns]
    if ausentes:
        raise ValueError(f"Campos obrigatórios ausentes: {ausentes}")
    # Mesma ordem de colunas do treino; campos extras são ignorados
    return df[FEATURES]


def prever(
    modelo: Pipeline,
    clientes: Cliente | Sequence[Cliente],
    limiar: float = LIMIAR_DECISAO,
) -> list[dict[str, Any]]:
    """Propensão de churn (0 a 1) e classe prevista para cada cliente.

    A propensão não é uma probabilidade calibrada: o modelo foi treinado com
    class_weight="balanced", o que desloca os valores para cima. Use-a para
    ordenar clientes e decidir com o limiar, não como taxa esperada de churn.
    """
    if not 0 < limiar < 1:
        raise ValueError(f"Limiar deve estar entre 0 e 1 (recebido: {limiar})")

    propensoes = modelo.predict_proba(montar_entrada(clientes))[:, 1]
    return [{"propensao_churn": float(p), "churn": bool(p >= limiar)} for p in propensoes]
