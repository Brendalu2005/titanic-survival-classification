"""Leitura e filtros do dashboard. Não treina modelos nem altera os resultados."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {0: "Não sobreviveu", 1: "Sobreviveu"}
MODEL_NAMES = {"logistica": "Regressão logística", "arvore": "Árvore", "xgboost": "XGBoost", "dummy": "Dummy"}


def load_evaluation(reports_dir):
    """Lê métricas já calculadas com respostas conhecidas no teste separado."""
    reports_dir = Path(reports_dir)
    try:
        metrics = json.loads((reports_dir / "metricas.json").read_text(encoding="utf-8"))
        for group in ["selected_threshold", "default_threshold", "dummy"]:
            for key in ["accuracy", "precision", "recall", "f1"]:
                value = float(metrics[group][key])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Métrica fora do intervalo esperado.")
        threshold = float(metrics["threshold"])
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        if not 0 <= threshold <= 1 or matrix.shape != (2, 2):
            raise ValueError("Limiar ou matriz de confusão inválidos.")
        if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix != np.floor(matrix)).any():
            raise ValueError("Contagens inválidas na matriz de confusão.")
        if matrix.sum() == 0:
            raise ValueError("A avaliação está vazia.")
        metrics["selected_model"]
    except FileNotFoundError as exc:
        raise ValueError("A avaliação ainda não foi gerada. Execute python -m src.train.") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Não foi possível ler a avaliação. Confira reports/metricas.json ou execute o treino novamente.") from exc
    return metrics


def load_predictions(prediction_path, passengers_path=None):
    """Acrescenta contexto dos passageiros quando o CSV de origem está disponível."""
    prediction_path = Path(prediction_path)
    if not prediction_path.is_file():
        raise ValueError("As previsões ainda não foram geradas. Execute python -m src.predict --input data/test.csv.")
    try:
        data = pd.read_csv(prediction_path)
        required = ["PassengerId", "ProbabilidadeSobrevivencia", "Survived"]
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError("Faltam colunas: " + ", ".join(missing))
        data = data[required].copy()
        for column in required:
            data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[required].to_numpy(dtype=float)).all():
            raise ValueError("Há valores ausentes ou infinitos nas previsões.")
        if data["PassengerId"].duplicated().any() or not (data["PassengerId"] == data["PassengerId"].round()).all():
            raise ValueError("Os identificadores devem ser inteiros e únicos.")
        if not data["ProbabilidadeSobrevivencia"].between(0, 1).all() or not data["Survived"].isin([0, 1]).all():
            raise ValueError("Probabilidades ou classes inválidas.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV de previsões inválido: {exc}") from exc
    data["PassengerId"] = data["PassengerId"].astype(int)
    data["Survived"] = data["Survived"].astype(int)
    if passengers_path is not None and Path(passengers_path).is_file():
        columns = ["PassengerId", "Name", "Sex", "Age", "Pclass"]
        try:
            passengers = pd.read_csv(passengers_path, usecols=lambda column: column in columns)
            if "PassengerId" in passengers and not passengers["PassengerId"].duplicated().any():
                data = data.merge(passengers, on="PassengerId", how="left", validate="one_to_one")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, TypeError, ValueError):
            # O contexto é opcional; as previsões continuam consultáveis sem ele.
            pass
    data["Previsao"] = data["Survived"].map(OUTCOMES)
    return data


def filter_predictions(data, outcomes, probability_range, search="", classes=None, sexes=None):
    """Filtra o lote de previsão; não recalcula métricas de avaliação."""
    low, high = probability_range
    mask = data["Previsao"].isin(outcomes) & data["ProbabilidadeSobrevivencia"].between(low / 100, high / 100)
    if search.strip():
        names = data["Name"].fillna("").astype(str) if "Name" in data else pd.Series("", index=data.index)
        mask &= (names.str.contains(search.strip(), case=False, regex=False)
                 | data["PassengerId"].astype(str).str.contains(search.strip(), regex=False))
    if classes is not None and "Pclass" in data:
        mask &= data["Pclass"].isin(classes)
    if sexes is not None and "Sex" in data:
        mask &= data["Sex"].isin(sexes)
    return data.loc[mask].sort_values("ProbabilidadeSobrevivencia", ascending=False).copy()
