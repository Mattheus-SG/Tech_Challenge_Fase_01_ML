"""Configuração central do projeto: sementes, caminhos e definição das features.

Tudo que o notebook 02 definia em células soltas fica aqui, em um único lugar (DRY).
"""

import os
from pathlib import Path

# ----- Reprodutibilidade -----
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
LIMIAR_DECISAO = 0.5

# ----- Caminhos (relativos à raiz do repositório, não ao diretório atual) -----
RAIZ = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ / "data" / "IBM_Telco_customer_churn_IBM_dataset.csv"
MODELS_DIR = RAIZ / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "02_modelo_campeao.joblib"))
AVALIACAO_PATH = MODELS_DIR / "02_avaliacao_campeao.json"


def caminho_metadata(caminho_modelo: Path) -> Path:
    """Os metadados ficam ao lado do modelo: modelo.joblib -> modelo_metadata.json."""
    return caminho_modelo.with_name(f"{caminho_modelo.stem}_metadata.json")


# ----- Modelo campeão (melhores hiperparâmetros do GridSearchCV da Etapa 2) -----
NOME_MODELO = "Random Forest (Tuning + KBest)"
K_FEATURES = "all"
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2,
    "class_weight": "balanced",
}

# ----- Colunas -----
TARGET = "Churn"

# Nomes originais do CSV -> nomes padronizados (sem espaços)
RENAME_MAP = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract": "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Value": TARGET,
}

COLUNAS_DESCARTADAS = {
    "vazamento (leakage)": ["Churn Label", "Churn Score", "CLTV", "Churn Reason"],
    "constantes": ["Count", "Country", "State"],
    "geografia": ["City", "Zip Code", "Lat Long", "Latitude", "Longitude"],
}

# Ordem igual à do treino: o scikit-learn exige as colunas na mesma ordem do fit
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CAT_FEATURES = [c for c in FEATURES if c not in NUM_FEATURES]
