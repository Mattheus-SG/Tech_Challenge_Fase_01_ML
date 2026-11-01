"""Carga e limpeza do dataset Telco (extraído do notebook 02_modelagem_e_avaliacao)."""

import logging
from pathlib import Path

import pandas as pd

from src.config import COLUNAS_DESCARTADAS, FEATURES, NUM_FEATURES, RENAME_MAP, TARGET

logger = logging.getLogger(__name__)


def carregar_dados(caminho: Path) -> pd.DataFrame:
    """Lê o CSV original e valida se as colunas esperadas estão presentes."""
    df = pd.read_csv(caminho)
    df.columns = df.columns.str.strip()

    esperadas = set(RENAME_MAP) | {c for cols in COLUNAS_DESCARTADAS.values() for c in cols}
    ausentes = esperadas - set(df.columns)
    if ausentes:
        raise KeyError(f"Colunas esperadas não encontradas no CSV: {sorted(ausentes)}")
    return df


def preparar_dados(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas inadequadas, padroniza nomes e trata valores ausentes."""
    df = df_raw.copy()

    # 1. Validações de integridade (antes de descartar as colunas usadas nelas)
    if not df["CustomerID"].is_unique:
        raise ValueError("CustomerID duplicado")
    if not ((df["Churn Label"] == "Yes") == (df["Churn Value"] == 1)).all():
        raise ValueError("Churn Label e Churn Value inconsistentes")

    # 2. Remove colunas de vazamento, constantes e geográficas
    descartar = [c for cols in COLUNAS_DESCARTADAS.values() for c in cols]
    df = df.drop(columns=descartar)

    # 3. Padroniza nomes e remove o identificador
    df = df.rename(columns=RENAME_MAP).drop(columns=["customerID"])

    # 4. SeniorCitizen: Yes/No -> 1/0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if df["SeniorCitizen"].isna().any():
        raise ValueError("Valor inesperado em SeniorCitizen (esperado Yes/No)")

    # 5. TotalCharges: texto -> número. Brancos pertencem a clientes com tenure == 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    sem_cobranca = df["TotalCharges"].isna()
    if (df.loc[sem_cobranca, "tenure"] != 0).any():
        raise ValueError("TotalCharges vazio em cliente com tenure > 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    logger.info("TotalCharges imputado com 0 em %d clientes (tenure = 0).", sem_cobranca.sum())

    # 6. Limpa espaços nas categorias
    for col in df.columns.difference(NUM_FEATURES + [TARGET]):
        df[col] = df[col].astype(str).str.strip()

    if df.isna().any().any():
        raise ValueError("Ainda existem valores nulos após a preparação")
    return df


def separar_features_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (na ordem de FEATURES) e y."""
    return df[FEATURES], df[TARGET].astype(int)
