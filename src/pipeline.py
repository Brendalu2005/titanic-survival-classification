"""Features e pré-processamento compartilhados por treino e previsão."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RAW_FEATURES = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
NUMERIC = ["Age", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin", "LogFare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]


def read_passengers(path, training=False):
    """Lê um CSV local e avisa quais dados obrigatórios estão faltando."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"CSV não encontrado: {path}. Coloque os dados na pasta data/.")
    data = pd.read_csv(path)
    required = RAW_FEATURES + ["PassengerId"] + (["Survived"] if training else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
    if data.empty:
        raise ValueError("O CSV não contém passageiros.")
    if data["PassengerId"].isna().any() or data["PassengerId"].duplicated().any():
        raise ValueError("PassengerId deve estar preenchido e não pode se repetir.")
    for column in ["Age", "Fare", "SibSp", "Parch", "Pclass"]:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"A coluna {column} deve conter números ou valores ausentes.") from exc
        if np.isinf(data[column]).any() or (data[column] < 0).any():
            raise ValueError(f"A coluna {column} não pode conter valores negativos ou infinitos.")
    if training and (data["Survived"].isna().any() or not data["Survived"].isin([0, 1]).all()):
        raise ValueError("Survived deve conter apenas 0 e 1, sem valores ausentes.")
    return data


def create_features(data):
    """Transformações por passageiro: não calculam estatísticas da amostra."""
    features = data[RAW_FEATURES].copy()
    title = features["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    features["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = (features["FamilySize"] == 1).astype(float)
    features.loc[features["FamilySize"].isna(), "IsAlone"] = np.nan
    features["HasCabin"] = features["Cabin"].notna().astype(int)
    features["LogFare"] = np.log1p(features["Fare"])
    # object + np.nan também funciona com os novos tipos de texto do pandas.
    for column in CATEGORICAL:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    return features[NUMERIC + CATEGORICAL]


def build_pipeline(model, scale=False):
    """Tudo que aprende com os dados fica dentro do Pipeline da validação."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparation = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(create_features, validate=False)),
        ("preparation", preparation),
        ("model", model),
    ])
