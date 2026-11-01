"""Schemas Pydantic da API: validam a entrada e documentam o contrato no Swagger (/docs)."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SimNao = Literal["Yes", "No"]
ServicoInternet = Literal["Yes", "No", "No internet service"]

SERVICOS_INTERNET = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Cliente de alto risco usado como exemplo no "Try it out" do Swagger
CLIENTE_EXEMPLO = {
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


class Cliente(BaseModel):
    """Dados de um cliente, nos mesmos nomes e categorias usados no treino."""

    model_config = ConfigDict(json_schema_extra={"examples": [CLIENTE_EXEMPLO]})

    gender: Literal["Female", "Male"] = Field(description="Gênero")
    SeniorCitizen: Literal[0, 1] = Field(description="1 se o cliente tem 65 anos ou mais")
    Partner: SimNao = Field(description="Tem cônjuge ou companheiro(a)")
    Dependents: SimNao = Field(description="Tem dependentes")
    tenure: int = Field(ge=0, description="Meses como cliente")
    PhoneService: SimNao = Field(description="Tem telefone fixo")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        description="Tem mais de uma linha"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(description="Tipo de internet")
    OnlineSecurity: ServicoInternet
    OnlineBackup: ServicoInternet
    DeviceProtection: ServicoInternet
    TechSupport: ServicoInternet
    StreamingTV: ServicoInternet
    StreamingMovies: ServicoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Tipo de contrato"
    )
    PaperlessBilling: SimNao = Field(description="Recebe fatura digital")
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(description="Forma de pagamento")
    MonthlyCharges: float = Field(
        ge=0, allow_inf_nan=False, description="Valor da mensalidade (US$)"
    )
    TotalCharges: float = Field(ge=0, allow_inf_nan=False, description="Total já cobrado (US$)")

    @model_validator(mode="after")
    def validar_combinacoes_de_servicos(self) -> Self:
        """Rejeita combinações que não existem nos dados (ex.: sem internet com TechSupport)."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(
                'MultipleLines deve ser "No phone service" se, e somente se, PhoneService for "No"'
            )
        sem_internet = self.InternetService == "No"
        for servico in SERVICOS_INTERNET:
            if sem_internet != (getattr(self, servico) == "No internet service"):
                raise ValueError(
                    f'{servico} deve ser "No internet service" se, e somente se, '
                    'InternetService for "No"'
                )
        return self


class Predicao(BaseModel):
    """Resultado do /predict."""

    propensao_churn: float = Field(
        ge=0, le=1, description="Propensão de churn (0 a 1). Não é probabilidade calibrada."
    )
    churn: bool = Field(description="True se a propensão atinge o limiar de decisão")
    limiar: float = Field(description="Limiar de decisão usado")
    versao_modelo: str = Field(description="Data de criação do modelo servido")


class Saude(BaseModel):
    """Resultado do /health."""

    status: Literal["healthy", "degraded"]
    modelo_carregado: bool
    modelo: str | None = Field(description="Nome do modelo servido")
    versao_modelo: str | None = Field(description="Data de criação do modelo servido")
    versao_api: str
