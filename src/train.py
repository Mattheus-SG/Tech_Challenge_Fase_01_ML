"""Treina o modelo campeão de ponta a ponta e salva o modelo e os metadados.

Uso: uv run python -m src.train --saida models/modelo_retreinado.joblib

A saída é obrigatória para não sobrescrever por engano o modelo oficial da Etapa 2
(models/02_modelo_campeao.joblib): o Random Forest retreinado em outra máquina é
equivalente, mas não idêntico bit a bit.
"""

import argparse
import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CAT_FEATURES,
    COLUNAS_DESCARTADAS,
    CV_FOLDS,
    DATA_PATH,
    LIMIAR_DECISAO,
    MODEL_PATH,
    NOME_MODELO,
    NUM_FEATURES,
    RF_PARAMS,
    SEED,
    TEST_SIZE,
    caminho_metadata,
)
from src.pipeline import criar_pipeline
from src.preprocessing import carregar_dados, preparar_dados, separar_features_alvo

logger = logging.getLogger(__name__)

METRICAS_CV = ["f1", "roc_auc", "precision", "recall"]


def validar_com_cv(modelo: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, float]:
    """Validação cruzada estratificada: média e desvio-padrão de cada métrica."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    res = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS_CV, n_jobs=-1)
    metricas = {}
    for nome in METRICAS_CV:
        metricas[f"cv_{nome}_media"] = float(res[f"test_{nome}"].mean())
        metricas[f"cv_{nome}_std"] = float(res[f"test_{nome}"].std())
    return metricas


def avaliar_no_teste(modelo: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Métricas no conjunto de teste (holdout)."""
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "teste_f1": float(f1_score(y, pred)),
        "teste_roc_auc": float(roc_auc_score(y, proba)),
        "teste_precision": float(precision_score(y, pred)),
        "teste_recall": float(recall_score(y, pred)),
        "teste_accuracy": float(accuracy_score(y, pred)),
    }


def treinar(
    caminho_modelo: Path,
    caminho_dados: Path = DATA_PATH,
    seed: int = SEED,
) -> tuple[Pipeline, dict]:
    """Carrega os dados, treina e avalia o pipeline e salva o modelo e os metadados."""
    X, y = separar_features_alvo(preparar_dados(carregar_dados(caminho_dados)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    modelo = criar_pipeline(seed)
    metricas = validar_com_cv(modelo, X_train, y_train, seed)
    modelo.fit(X_train, y_train)
    metricas |= avaliar_no_teste(modelo, X_test, y_test)

    # Mesma estrutura do 02_modelo_campeao_metadata.json gerado pelo notebook da Etapa 2
    metadata = {
        "modelo": NOME_MODELO,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "criterio_de_escolha": "maior F1 médio na validação cruzada (desempate: AUC-ROC)",
        "seed": seed,
        "cv_folds": CV_FOLDS,
        "test_size": TEST_SIZE,
        "dataset": Path(caminho_dados).name,
        "n_linhas": int(len(X)),
        "taxa_churn": float(y.mean()),
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "features_numericas": NUM_FEATURES,
        "features_categoricas": {c: sorted(X[c].unique().tolist()) for c in CAT_FEATURES},
        "colunas_descartadas": COLUNAS_DESCARTADAS,
        "metricas": metricas,
        "hiperparametros": RF_PARAMS,
        "limiar_decisao": LIMIAR_DECISAO,
    }

    caminho_modelo = Path(caminho_modelo)
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho_modelo)
    caminho_metadata(caminho_modelo).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Modelo salvo em %s", caminho_modelo)
    return modelo, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo campeão de churn.")
    parser.add_argument("--saida", type=Path, required=True, help="caminho do .joblib a gerar")
    args = parser.parse_args()
    if args.saida.resolve() == MODEL_PATH.resolve():
        parser.error(f"{MODEL_PATH} é o modelo oficial servido pela API; escolha outro caminho")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, metadata = treinar(args.saida)
    for nome, valor in metadata["metricas"].items():
        logger.info("%s: %.4f", nome, valor)


if __name__ == "__main__":
    main()
