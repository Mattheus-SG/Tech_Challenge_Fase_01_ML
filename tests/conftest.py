"""Fixtures compartilhadas pelos testes."""

import pandas as pd
import pytest

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
