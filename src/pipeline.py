"""Pipeline do modelo campeão: pré-processamento, seleção de atributos e Random Forest."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CAT_FEATURES, K_FEATURES, NUM_FEATURES, RF_PARAMS, SEED


def criar_preprocessor() -> ColumnTransformer:
    """Padroniza as features numéricas e aplica one-hot nas categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def criar_pipeline(seed: int = SEED) -> Pipeline:
    """Pipeline completo, ainda não treinado, com os hiperparâmetros escolhidos na Etapa 2."""
    return Pipeline(
        steps=[
            ("preprocessor", criar_preprocessor()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=K_FEATURES)),
            ("classifier", RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=-1)),
        ]
    )
