"""Testes de src/preprocessing.py."""

import json

import pandas as pd
import pytest

from src.config import COLUNAS_DESCARTADAS, DATA_PATH, FEATURES, MODELS_DIR, TARGET
from src.preprocessing import carregar_dados, preparar_dados, separar_features_alvo


def test_preparar_dados_remove_vazamento_e_identificador(df_bruto):
    df = preparar_dados(df_bruto)

    descartadas = {c for cols in COLUNAS_DESCARTADAS.values() for c in cols}
    assert descartadas.isdisjoint(df.columns)
    assert "customerID" not in df.columns
    assert set(df.columns) == set(FEATURES) | {TARGET}


@pytest.mark.parametrize("valor, esperado", [("Yes", 1), ("No", 0), (" Yes ", 1)])
def test_preparar_dados_converte_senior_citizen(df_bruto, valor, esperado):
    df_bruto["Senior Citizen"] = valor

    df = preparar_dados(df_bruto)

    assert (df["SeniorCitizen"] == esperado).all()


def test_preparar_dados_imputa_zero_em_total_charges_de_cliente_novo(df_bruto):
    df = preparar_dados(df_bruto)

    assert df.loc[2, "TotalCharges"] == 0.0
    assert df["TotalCharges"].dtype == float


def test_preparar_dados_remove_espacos_das_categorias(df_bruto):
    df = preparar_dados(df_bruto)

    assert df.loc[2, "Contract"] == "Two year"


def test_preparar_dados_nao_altera_dataframe_original(df_bruto):
    original = df_bruto.copy()

    preparar_dados(df_bruto)

    pd.testing.assert_frame_equal(df_bruto, original)


def test_preparar_dados_falha_com_total_charges_vazio_e_tenure_positivo(df_bruto):
    df_bruto.loc[0, "Total Charges"] = " "

    with pytest.raises(ValueError, match="tenure > 0"):
        preparar_dados(df_bruto)


def test_preparar_dados_falha_com_customer_id_duplicado(df_bruto):
    df_bruto.loc[1, "CustomerID"] = df_bruto.loc[0, "CustomerID"]

    with pytest.raises(ValueError, match="duplicado"):
        preparar_dados(df_bruto)


def test_carregar_dados_falha_sem_coluna_esperada(df_bruto, tmp_path):
    caminho = tmp_path / "incompleto.csv"
    df_bruto.drop(columns=["Contract"]).to_csv(caminho, index=False)

    with pytest.raises(KeyError, match="Contract"):
        carregar_dados(caminho)


def test_separar_features_alvo_mantem_ordem_do_treino(df_bruto):
    X, y = separar_features_alvo(preparar_dados(df_bruto))

    assert list(X.columns) == FEATURES
    assert y.tolist() == [1, 0, 1]


def test_preparar_dados_no_dataset_real_reproduz_o_notebook():
    """Teste de caracterização: a refatoração gera os mesmos dados que o notebook 02."""
    metadata = json.loads((MODELS_DIR / "02_modelo_campeao_metadata.json").read_text())

    X, y = separar_features_alvo(preparar_dados(carregar_dados(DATA_PATH)))

    assert len(X) == metadata["n_linhas"]
    assert y.mean() == pytest.approx(metadata["taxa_churn"])
    for coluna, categorias in metadata["features_categoricas"].items():
        assert sorted(X[coluna].unique()) == categorias
