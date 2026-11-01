"""Testes da API (src/api) com o TestClient do FastAPI."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.api.main import ModeloServido, app, obter_modelo
from src.api.schemas import SERVICOS_INTERNET
from src.predict import prever


@pytest.fixture
def client():
    """Cliente HTTP de teste; o `with` executa o lifespan (carga do modelo)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_sem_modelo(monkeypatch, tmp_path):
    """API subindo com um MODEL_PATH inexistente."""
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nao_existe.joblib")
    api.carregar_modelo_servido.cache_clear()
    with TestClient(app) as c:
        yield c
    api.carregar_modelo_servido.cache_clear()


class ModeloQuebrado:
    """Dublê de modelo que falha na inferência com uma mensagem interna."""

    def predict_proba(self, X):
        raise RuntimeError("detalhe interno: /srv/segredo.joblib")


# ----- /health -----


def test_health_responde_healthy_com_modelo_carregado(client):
    resposta = client.get("/health")

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "healthy"
    assert corpo["modelo_carregado"] is True
    assert corpo["versao_modelo"]


def test_health_responde_degraded_sem_modelo(client_sem_modelo):
    resposta = client_sem_modelo.get("/health")

    assert resposta.status_code == 503
    assert resposta.json()["status"] == "degraded"
    assert resposta.json()["modelo_carregado"] is False


# ----- /predict: sucesso -----


def test_predict_devolve_propensao_e_classe(client, cliente):
    resposta = client.post("/predict", json=cliente)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"propensao_churn", "churn", "limiar", "versao_modelo"}
    assert 0 <= corpo["propensao_churn"] <= 1
    assert corpo["churn"] is True


def test_predict_da_o_mesmo_resultado_que_o_modulo_de_inferencia(client, cliente, modelo):
    resposta = client.post("/predict", json=cliente)

    [esperado] = prever(modelo, cliente)
    assert resposta.json()["propensao_churn"] == pytest.approx(esperado["propensao_churn"])


def test_predict_classifica_cliente_fiel_como_nao_churn(client, cliente_fiel):
    resposta = client.post("/predict", json=cliente_fiel)

    assert resposta.status_code == 200
    assert resposta.json()["churn"] is False


def test_predict_aceita_cliente_sem_internet(client, cliente):
    cliente["InternetService"] = "No"
    for servico in SERVICOS_INTERNET:
        cliente[servico] = "No internet service"

    resposta = client.post("/predict", json=cliente)

    assert resposta.status_code == 200


def test_exemplo_do_swagger_e_um_cliente_valido(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Cliente"]
    [exemplo] = schema["examples"]

    resposta = client.post("/predict", json=exemplo)

    assert resposta.status_code == 200


def test_resposta_informa_o_tempo_de_processamento(client):
    resposta = client.get("/health")

    assert float(resposta.headers["X-Process-Time"]) >= 0


def test_raiz_redireciona_para_a_documentacao(client):
    resposta = client.get("/", follow_redirects=False)

    assert resposta.status_code == 307
    assert resposta.headers["location"] == "/docs"


# ----- /predict: erros -----


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("Contract", "Three year"),
        ("gender", "F"),
        ("SeniorCitizen", 2),
        ("tenure", -1),
        ("MonthlyCharges", "caro"),
    ],
)
def test_predict_rejeita_valor_invalido_com_422(client, cliente, campo, valor):
    cliente[campo] = valor

    resposta = client.post("/predict", json=cliente)

    assert resposta.status_code == 422
    assert campo in resposta.text


@pytest.mark.parametrize("campo", ["MonthlyCharges", "TotalCharges"])
def test_predict_rejeita_valor_infinito_com_422(client, cliente, campo):
    cliente[campo] = 0.0
    corpo = json.dumps(cliente).replace(f'"{campo}": 0.0', f'"{campo}": 1e999')

    resposta = client.post("/predict", content=corpo, headers={"Content-Type": "application/json"})

    assert resposta.status_code == 422


def test_predict_rejeita_campo_ausente_com_422(client, cliente):
    del cliente["tenure"]

    resposta = client.post("/predict", json=cliente)

    assert resposta.status_code == 422


@pytest.mark.parametrize(
    "alteracoes",
    [
        {"InternetService": "No"},  # sem internet, mas com serviços "No"
        {"OnlineSecurity": "No internet service"},  # tem internet
        {"PhoneService": "No"},  # sem telefone, mas MultipleLines "No"
        {"MultipleLines": "No phone service"},  # tem telefone
    ],
)
def test_predict_rejeita_combinacao_impossivel_de_servicos(client, cliente, alteracoes):
    resposta = client.post("/predict", json=cliente | alteracoes)

    assert resposta.status_code == 422


def test_predict_responde_503_sem_modelo(client_sem_modelo, cliente):
    resposta = client_sem_modelo.post("/predict", json=cliente)

    assert resposta.status_code == 503


def test_predict_responde_500_generico_sem_expor_detalhes(client, cliente):
    app.dependency_overrides[obter_modelo] = lambda: ModeloServido(
        ModeloQuebrado(), nome="quebrado", versao="0"
    )
    try:
        resposta = client.post("/predict", json=cliente)
    finally:
        app.dependency_overrides.clear()

    assert resposta.status_code == 500
    assert "segredo" not in resposta.text
