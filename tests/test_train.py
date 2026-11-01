"""Testes de src/train.py (treinam o modelo completo: marcados como slow)."""

import json

import joblib
import numpy as np
import pytest

from src.config import MODELS_DIR, caminho_metadata
from src.train import treinar

pytestmark = pytest.mark.slow

# Métricas registradas pelo notebook da Etapa 2. O Random Forest não é bit a bit idêntico
# entre máquinas (CPU/ponto flutuante), por isso a comparação usa tolerância.
METADATA_ETAPA_2 = MODELS_DIR / "02_modelo_campeao_metadata.json"
TOLERANCIA = 0.01


@pytest.fixture(scope="module")
def treino(tmp_path_factory):
    """Treina uma vez em uma pasta temporária (nunca sobrescreve models/)."""
    caminho = tmp_path_factory.mktemp("models") / "modelo.joblib"
    modelo, metadata = treinar(caminho_modelo=caminho)
    return caminho, modelo, metadata


def test_treinar_salva_modelo_e_metadata(treino):
    caminho, _, metadata = treino

    assert caminho.exists()
    assert json.loads(caminho_metadata(caminho).read_text(encoding="utf-8")) == metadata


def test_modelo_salvo_recarrega_com_as_mesmas_predicoes(treino, dados_reais):
    caminho, modelo, _ = treino
    X, _ = dados_reais

    recarregado = joblib.load(caminho)

    np.testing.assert_allclose(recarregado.predict_proba(X), modelo.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize(
    "metrica", ["cv_f1_media", "cv_roc_auc_media", "teste_f1", "teste_roc_auc", "teste_recall"]
)
def test_treinar_reproduz_as_metricas_da_etapa_2(treino, metrica):
    _, _, metadata = treino
    referencia = json.loads(METADATA_ETAPA_2.read_text(encoding="utf-8"))["metricas"]

    assert metadata["metricas"][metrica] == pytest.approx(referencia[metrica], abs=TOLERANCIA)


def test_treinar_concorda_com_o_modelo_da_etapa_2(treino, dados_reais):
    """Caracterização: o modelo retreinado classifica os clientes como o .joblib da Etapa 2."""
    _, modelo, _ = treino
    X, _ = dados_reais
    modelo_etapa_2 = joblib.load(MODELS_DIR / "02_modelo_campeao.joblib")

    concordancia = (modelo.predict(X) == modelo_etapa_2.predict(X)).mean()

    assert concordancia >= 0.99
