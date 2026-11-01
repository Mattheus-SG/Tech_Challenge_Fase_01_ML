"""Fixtures compartilhadas pelos testes."""

import pandas as pd
import pytest

from src.config import DATA_PATH
from src.predict import carregar_modelo
from src.preprocessing import carregar_dados, preparar_dados, separar_features_alvo

# Um cliente no formato original do CSV (33 colunas)
CLIENTE_BRUTO = {
    "CustomerID": "0001-AAAAA",
    "Count": 1,
    "Country": "United States",
    "State": "California",
    "City": "Los Angeles",
    "Zip Code": 90003,
    "Lat Long": "33.964131, -118.272783",
    "Latitude": 33.964131,
    "Longitude": -118.272783,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Mailed check",
    "Monthly Charges": 53.85,
    "Total Charges": "108.15",
    "Churn Label": "Yes",
    "Churn Value": 1,
    "Churn Score": 86,
    "CLTV": 3239,
    "Churn Reason": "Competitor made better offer",
}


@pytest.fixture
def df_bruto() -> pd.DataFrame:
    """Três clientes sintéticos: um comum, um sênior e um novo (tenure 0, sem cobrança)."""
    senior = {
        **CLIENTE_BRUTO,
        "CustomerID": "0002-BBBBB",
        "Senior Citizen": "Yes",
        "Churn Label": "No",
        "Churn Value": 0,
    }
    novo = {
        **CLIENTE_BRUTO,
        "CustomerID": "0003-CCCCC",
        "Tenure Months": 0,
        "Total Charges": " ",
        "Contract": " Two year ",
    }
    return pd.DataFrame([CLIENTE_BRUTO, senior, novo])


@pytest.fixture(scope="session")
def dados_reais() -> tuple[pd.DataFrame, pd.Series]:
    """X e y do dataset real, já preparados (carregados uma vez por execução)."""
    return separar_features_alvo(preparar_dados(carregar_dados(DATA_PATH)))


# Clientes no formato de entrada do modelo (o mesmo que a API vai receber)
CLIENTE_ALTO_RISCO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}

CLIENTE_FIEL = {
    **CLIENTE_ALTO_RISCO,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 65.0,
    "TotalCharges": 3900.0,
}


@pytest.fixture
def cliente() -> dict:
    """Cópia do cliente de alto risco (cada teste pode alterá-la à vontade)."""
    return dict(CLIENTE_ALTO_RISCO)


@pytest.fixture
def cliente_fiel() -> dict:
    """Cópia de um cliente de perfil fiel (contrato longo, serviços de suporte)."""
    return dict(CLIENTE_FIEL)


@pytest.fixture(scope="session")
def modelo():
    """Modelo oficial da Etapa 2, carregado uma vez por execução."""
    return carregar_modelo()
