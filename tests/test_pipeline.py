"""Testes de src/pipeline.py."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.config import RF_PARAMS, SEED
from src.pipeline import criar_pipeline


@pytest.fixture(scope="module")
def amostra(dados_reais):
    """Amostra estratificada pequena (o CSV vem ordenado por churn: não use as primeiras linhas)."""
    X, y = dados_reais
    return train_test_split(X, y, train_size=1000, test_size=300, random_state=SEED, stratify=y)


def test_criar_pipeline_tem_as_etapas_do_modelo_campeao():
    pipeline = criar_pipeline()

    assert list(pipeline.named_steps) == ["preprocessor", "feature_selection", "classifier"]
    params_rf = pipeline.named_steps["classifier"].get_params()
    assert {k: params_rf[k] for k in RF_PARAMS} == RF_PARAMS


def test_criar_pipeline_gera_probabilidades_validas(amostra):
    X_train, X_test, y_train, _ = amostra

    proba = criar_pipeline().fit(X_train, y_train).predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_criar_pipeline_e_reprodutivel_com_a_mesma_seed(amostra):
    X_train, X_test, y_train, _ = amostra

    proba_1 = criar_pipeline(seed=7).fit(X_train, y_train).predict_proba(X_test)
    proba_2 = criar_pipeline(seed=7).fit(X_train, y_train).predict_proba(X_test)

    np.testing.assert_allclose(proba_1, proba_2, atol=1e-12)


def test_criar_pipeline_muda_com_outra_seed(amostra):
    X_train, X_test, y_train, _ = amostra

    proba_1 = criar_pipeline(seed=1).fit(X_train, y_train).predict_proba(X_test)
    proba_2 = criar_pipeline(seed=2).fit(X_train, y_train).predict_proba(X_test)

    assert not np.allclose(proba_1, proba_2)
