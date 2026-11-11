"""Referência inicial avaliada somente por validação dentro dos 80% de treino."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from src.pipeline import RAW_FEATURES, ROOT, SEED, build_pipeline, read_passengers
from src.train import SCORING, choose_threshold


def run_baseline(csv_path=ROOT / "data/train.csv"):
    """Não avalia o teste, não salva modelo e não altera os relatórios finais.

    O F1 usado para escolher o limiar é uma medida de seleção interna, não uma
    estimativa independente do desempenho do limiar escolhido em novos dados.
    """
    data = read_passengers(csv_path, training=True)
    train, _ = train_test_split(
        data, test_size=0.2, stratify=data["Survived"], random_state=SEED,
    )
    X, y = train[RAW_FEATURES], train["Survived"]
    if y.value_counts().min() < 5:
        raise ValueError("O treino precisa de pelo menos 5 passageiros de cada classe para a validação.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipelines = {
        "dummy": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "logistica": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=SEED), scale=True,
        ),
    }
    rows = []
    for name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, error_score="raise")
        rows.append({"model": name, **{
            f"cv_{metric}": float(scores[f"test_{metric}"].mean()) for metric in SCORING
        }})
    probabilities = cross_val_predict(
        pipelines["logistica"], X, y, cv=cv, method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold, thresholds = choose_threshold(y, probabilities)
    return {
        "comparison": pd.DataFrame(rows), "threshold": threshold,
        "thresholds": thresholds, "train_rows": len(train),
        "note": "Validação interna nos 80% de treino; o teste não foi avaliado nesta execução. "
                "As métricas usadas para escolher o limiar não são uma avaliação independente.",
    }
