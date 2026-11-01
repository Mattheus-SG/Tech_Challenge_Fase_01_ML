"""Testes de src/evaluate.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import AVALIACAO_PATH, MODELS_DIR
from src.evaluate import GRUPOS, avaliar, cobertura_cobranca, desempenho_do_grupo


@pytest.fixture(scope="module")
def avaliacao():
    return avaliar()


def test_cobertura_cobranca_considera_so_quem_cancelou():
    mensalidades = pd.Series([100.0, 50.0, 50.0, 999.0])
    y = pd.Series([1, 1, 1, 0])
    pred = np.array([1, 0, 1, 1])

    assert cobertura_cobranca(mensalidades, y, pred) == pytest.approx(0.75)


def test_desempenho_do_grupo_calcula_alarmes_falsos_entre_quem_permaneceu():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.2, 0.8, 0.1])

    grupo = desempenho_do_grupo(y, pred, proba)

    assert grupo["n"] == 4
    assert grupo["recall"] == 0.5
    assert grupo["alarmes_falsos"] == 0.5


def test_avaliacao_cobre_todos_os_grupos_e_os_dois_lados_do_conjunto_de_teste(avaliacao):
    grupos = avaliacao["grupos"]

    assert set(grupos) == set(GRUPOS)
    assert grupos["Mulheres"]["n"] + grupos["Homens"]["n"] == 1409


def test_avaliacao_reproduz_a_matriz_de_confusao_do_modelo_oficial(avaliacao):
    assert avaliacao["matriz_confusao_teste"] == {"tn": 787, "fp": 248, "fn": 83, "tp": 291}


def test_campeao_supera_o_baseline_com_os_mesmos_pesos(avaliacao):
    """Caracterização: a vantagem existe, mas é pequena (ver Model Card, seção 4.1)."""
    metadata = json.loads((MODELS_DIR / "02_modelo_campeao_metadata.json").read_text("utf-8"))

    baseline = avaliacao["baseline_balanceado"]

    assert metadata["metricas"]["cv_f1_media"] > baseline["cv_f1_media"]
    assert metadata["metricas"]["teste_f1"] > baseline["teste_f1"]


def test_arquivo_salvo_esta_atualizado(avaliacao):
    salvo = json.loads(AVALIACAO_PATH.read_text(encoding="utf-8"))

    assert salvo["metrica_negocio"] == avaliacao["metrica_negocio"]
    assert salvo["grupos"] == avaliacao["grupos"]
