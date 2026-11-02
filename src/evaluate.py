"""Avaliação complementar do modelo oficial: gera os números do Model Card que não vêm do treino.

Uso: uv run python -m src.evaluate

Não treina nem altera o modelo oficial. Calcula, no mesmo conjunto de teste da Etapa 2:
a métrica de negócio, o desempenho por subgrupo, a diferença entre treino e teste e a
comparação com um baseline que usa o mesmo class_weight="balanced" do campeão. Salva tudo em
models/02_avaliacao_campeao.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    AVALIACAO_PATH,
    DATA_PATH,
    LIMIAR_DECISAO,
    MODEL_PATH,
    SEED,
    TEST_SIZE,
    caminho_metadata,
)
from src.pipeline import criar_preprocessor
from src.predict import carregar_modelo
from src.preprocessing import carregar_dados, preparar_dados, separar_features_alvo
from src.train import avaliar_no_teste, validar_com_cv

logger = logging.getLogger(__name__)

# Subgrupos da seção 5 do Model Card
GRUPOS = {
    "Mulheres": lambda X: X["gender"] == "Female",
    "Homens": lambda X: X["gender"] == "Male",
    "Menos de 65 anos": lambda X: X["SeniorCitizen"] == 0,
    "65 anos ou mais": lambda X: X["SeniorCitizen"] == 1,
    "Sem parceiro(a)": lambda X: X["Partner"] == "No",
    "Com parceiro(a)": lambda X: X["Partner"] == "Yes",
    "Sem dependentes": lambda X: X["Dependents"] == "No",
    "Com dependentes": lambda X: X["Dependents"] == "Yes",
    "Tenure acima de 48 meses": lambda X: X["tenure"] > 48,
}


def criar_baseline_balanceado(seed: int = SEED) -> Pipeline:
    """Regressão Logística com o mesmo class_weight do campeão, para comparação justa."""
    return Pipeline(
        steps=[
            ("preprocessor", criar_preprocessor()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    )


def cobertura_cobranca(mensalidades: pd.Series, y: pd.Series, pred: np.ndarray) -> float:
    """Parcela da cobrança mensal dos clientes que cancelaram que o modelo sinalizou."""
    cancelou = y.to_numpy() == 1
    valores = mensalidades.to_numpy()
    return float(valores[cancelou & (pred == 1)].sum() / valores[cancelou].sum())


def desempenho_do_grupo(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    """Métricas de um subgrupo; alarmes_falsos é a taxa entre os que permaneceram."""
    return {
        "n": int(len(y)),
        "churn_real": float(y.mean()),
        "churners": int(y.sum()),
        "sinalizados": float(pred.mean()),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "alarmes_falsos": float(pred[y == 0].mean()),
        "auc": float(roc_auc_score(y, proba)),
    }


def avaliar(caminho_dados: Path = DATA_PATH, seed: int = SEED) -> dict:
    """Calcula a avaliação complementar usando o modelo oficial e a divisão da Etapa 2."""
    X, y = separar_features_alvo(preparar_dados(carregar_dados(caminho_dados)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    modelo = carregar_modelo()
    proba = modelo.predict_proba(X_test)[:, 1]
    pred = (proba >= LIMIAR_DECISAO).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    proba_treino = modelo.predict_proba(X_train)[:, 1]

    baseline = criar_baseline_balanceado(seed)
    metricas_baseline = validar_com_cv(baseline, X_train, y_train, seed)
    baseline.fit(X_train, y_train)
    metricas_baseline |= avaliar_no_teste(baseline, X_test, y_test)
    fp_baseline = int(((baseline.predict(X_test) == 1) & (y_test.to_numpy() == 0)).sum())

    grupos = {}
    for nome, filtro in GRUPOS.items():
        mascara = filtro(X_test).to_numpy()
        grupos[nome] = desempenho_do_grupo(
            y_test.to_numpy()[mascara], pred[mascara], proba[mascara]
        )

    metadata = json.loads(caminho_metadata(MODEL_PATH).read_text(encoding="utf-8"))

    return {
        "versao_modelo": metadata["criado_em"],
        "limiar_decisao": LIMIAR_DECISAO,
        "matriz_confusao_teste": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrica_negocio": {
            "cobertura_cobranca_churn": cobertura_cobranca(X_test["MonthlyCharges"], y_test, pred),
            "clientes_sinalizados": int(pred.sum()),
            "clientes_no_teste": int(len(y_test)),
        },
        "treino_vs_teste": {
            "auc_treino": float(roc_auc_score(y_train, proba_treino)),
            "auc_teste": float(roc_auc_score(y_test, proba)),
            "f1_treino": float(f1_score(y_train, proba_treino >= LIMIAR_DECISAO)),
            "f1_teste": float(f1_score(y_test, pred)),
        },
        "baseline_balanceado": {**metricas_baseline, "alarmes_falsos_teste": fp_baseline},
        "grupos": grupos,
    }


def salvar_avaliacao() -> None:
    """Avalia o modelo oficial e grava o resultado em models/02_avaliacao_campeao.json."""
    AVALIACAO_PATH.write_text(
        json.dumps(avaliar(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    salvar_avaliacao()
    logger.info("Avaliação salva em %s", AVALIACAO_PATH)


if __name__ == "__main__":
    main()
