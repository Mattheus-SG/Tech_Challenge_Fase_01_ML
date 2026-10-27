"""Testes de src/predict.py."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.config import FEATURES, SEED
from src.predict import carregar_modelo, montar_entrada, prever


def test_carregar_modelo_devolve_pipeline_treinado(modelo):
    assert list(modelo.feature_names_in_) == FEATURES
    assert hasattr(modelo, "predict_proba")


def test_carregar_modelo_falha_com_caminho_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        carregar_modelo(tmp_path / "nao_existe.joblib")


def test_montar_entrada_aceita_um_cliente_ou_uma_lista(cliente):
    assert montar_entrada(cliente).shape == (1, len(FEATURES))
    assert montar_entrada([cliente, cliente]).shape == (2, len(FEATURES))


def test_montar_entrada_reordena_colunas_e_ignora_campos_extras(cliente):
    fora_de_ordem = dict(reversed(list(cliente.items()))) | {"customerID": "0001-AAAAA"}

    df = montar_entrada(fora_de_ordem)

    assert list(df.columns) == FEATURES


def test_montar_entrada_falha_sem_campo_obrigatorio(cliente):
    del cliente["tenure"]

    with pytest.raises(ValueError, match="tenure"):
        montar_entrada(cliente)


def test_montar_entrada_falha_sem_clientes():
    with pytest.raises(ValueError, match="Nenhum cliente"):
        montar_entrada([])


def test_prever_devolve_propensao_e_classe(modelo, cliente):
    [resultado] = prever(modelo, cliente)

    assert set(resultado) == {"propensao_churn", "churn"}
    assert 0 <= resultado["propensao_churn"] <= 1
    assert isinstance(resultado["churn"], bool)


def test_prever_cliente_de_alto_risco_tem_propensao_maior_que_cliente_fiel(
    modelo, cliente, cliente_fiel
):
    """Teste direcional: o perfil clássico de churn deve ficar acima do perfil fiel."""
    risco, fiel = prever(modelo, [cliente, cliente_fiel])

    assert risco["churn"] is True
    assert fiel["churn"] is False
    assert risco["propensao_churn"] > fiel["propensao_churn"]


@pytest.mark.parametrize("limiar, esperado", [(0.01, True), (0.99, False)])
def test_prever_respeita_o_limiar(modelo, cliente, limiar, esperado):
    [resultado] = prever(modelo, cliente, limiar=limiar)

    assert resultado["churn"] is esperado


@pytest.mark.parametrize("limiar", [0, 1, -0.1, 1.5])
def test_prever_falha_com_limiar_invalido(modelo, cliente, limiar):
    with pytest.raises(ValueError, match="Limiar"):
        prever(modelo, cliente, limiar=limiar)


def test_prever_concorda_com_o_predict_do_modelo(modelo, dados_reais):
    """O caminho dict -> DataFrame da API dá o mesmo resultado que o DataFrame do treino."""
    X, y = dados_reais
    _, amostra = train_test_split(X, test_size=500, random_state=SEED, stratify=y)

    resultados = prever(modelo, amostra.to_dict("records"))

    np.testing.assert_allclose(
        [r["propensao_churn"] for r in resultados], modelo.predict_proba(amostra)[:, 1]
    )
    assert [r["churn"] for r in resultados] == modelo.predict(amostra).astype(bool).tolist()
