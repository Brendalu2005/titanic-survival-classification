"""Treinamento reproduzível e registro local: python -m src.train."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Também gera gráficos em terminais sem interface gráfica.
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.pipeline import RAW_FEATURES, ROOT, SEED, build_pipeline, read_passengers

SCORING = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0), "recall": make_scorer(recall_score, zero_division=0), "f1": "f1", "roc_auc": "roc_auc"}
TEST_NOTE = "Avaliação didática em conjunto previamente consultado no baseline; não é uma estimativa independente."


def model_specs():
    """Grades pequenas para estudar sem transformar o projeto em uma busca extensa."""
    return {
        "dummy": (DummyClassifier(strategy="most_frequent"), {}, False),
        "logistica": (LogisticRegression(max_iter=1000, random_state=SEED), {"model__C": [0.1, 1, 10]}, True),
        "arvore": (DecisionTreeClassifier(random_state=SEED), {
            "model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 5],
        }, False),
        "xgboost": (XGBClassifier(
            learning_rate=0.1, random_state=SEED, n_jobs=1,
            tree_method="hist", device="cpu", eval_metric="logloss",
        ), {"model__max_depth": [2, 3], "model__n_estimators": [100, 200]}, False),
    }


def evaluate(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def choose_threshold(y, probabilities):
    rows = [{"threshold": round(float(t), 2), **evaluate(y, probabilities, t)}
            for t in np.round(np.arange(0.20, 0.801, 0.05), 2)]
    table = pd.DataFrame(rows)
    # Arredondar evita que diferenças de representação mudem os desempates.
    ranking = table.assign(distance=(table["threshold"] - 0.5).abs().round(8))
    ranking = ranking.sort_values(["f1", "distance", "threshold"], ascending=[False, True, False])
    return float(ranking.iloc[0]["threshold"]), table


def compare_models(X, y, cv, reports, metadata):
    comparison, searches = [], {}
    for name, (model, grid, scale) in model_specs().items():
        print(f"Validação cruzada: {name}", flush=True)
        search = GridSearchCV(
            build_pipeline(model, scale), grid, scoring=SCORING, refit="f1",
            cv=cv, return_train_score=True, n_jobs=1, error_score="raise",
        ).fit(X, y)
        searches[name] = search
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(reports / f"busca_{name}.csv", index=False)
        best = search.best_index_
        row = {"model": name, "parameters": json.dumps(search.best_params_)}
        for metric in SCORING:
            row[f"cv_{metric}"] = float(results.loc[best, f"mean_test_{metric}"])
            row[f"cv_{metric}_std"] = float(results.loc[best, f"std_test_{metric}"])
        row["train_f1"] = float(results.loc[best, "mean_train_f1"])
        row["train_accuracy"] = float(results.loc[best, "mean_train_accuracy"])
        comparison.append(row)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"stage": "model_selection", "data_sha256": metadata["data_sha256"]})
            mlflow.log_params({"seed": SEED, "folds": 5, **search.best_params_})
            mlflow.log_metrics({key: value for key, value in row.items() if isinstance(value, float)})
            mlflow.log_dict(metadata, "metadata.json")
            mlflow.log_artifact(str(reports / f"busca_{name}.csv"))
    table = pd.DataFrame(comparison)
    table.to_csv(reports / "comparacao_modelos.csv", index=False)
    priority = {"logistica": 0, "arvore": 1, "xgboost": 2}
    candidates = table[table["model"] != "dummy"].copy()
    candidates["priority"] = candidates["model"].map(priority)
    candidates["score"] = candidates["cv_f1"].round(12)
    winner = candidates.sort_values(["score", "priority"], ascending=[False, True]).iloc[0]["model"]
    return winner, searches, table


def save_plots(comparison, thresholds, importance, y_test, probabilities, threshold, reports):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(comparison["model"], comparison["cv_f1"], yerr=comparison["cv_f1_std"], capsize=4)
    axes[0].set(title="F1 na validação cruzada (média e desvio)", ylabel="F1", ylim=(0, 1))
    for metric in ["precision", "recall", "f1"]:
        axes[1].plot(thresholds["threshold"], thresholds[metric], marker=".", label=metric)
    axes[1].axvline(threshold, color="gray", linestyle="--")
    axes[1].set(title="Ajuste interno com previsões fora da partição", xlabel="Limiar", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(reports / "validacao.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, probabilities >= threshold, labels=[0, 1],
        display_labels=["Não sobreviveu", "Sobreviveu"], cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title("Matriz de confusão — avaliação didática\nTeste previamente consultado", fontsize=11)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe observada")
    fig.tight_layout()
    fig.savefig(reports / "matriz_confusao.png", dpi=140)
    plt.close(fig)

    ordered = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set(title="Permutação no teste: interpretação posterior", xlabel="Redução do F1 com limiar 0,50")
    fig.tight_layout()
    fig.savefig(reports / "importancia.png", dpi=140)
    plt.close(fig)


def experiment_metadata(csv_path):
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "mlflow", "joblib"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "indisponivel", True
    return {
        "seed": SEED, "test_size": 0.2, "folds": 5, "data_file": str(Path(csv_path).resolve()),
        "data_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "versions": {package: importlib.metadata.version(package) for package in packages},
        "git_commit": revision, "git_dirty": dirty, "test_note": TEST_NOTE,
    }


def run_training(csv_path=ROOT / "data/train.csv", output_dir=ROOT):
    """O notebook e a linha de comando executam exatamente este fluxo."""
    data = read_passengers(csv_path, training=True)
    train, test = train_test_split(data, test_size=0.2, stratify=data["Survived"], random_state=SEED)
    if train["Survived"].value_counts().min() < 5:
        raise ValueError("O treino precisa de pelo menos 5 passageiros de cada classe para a validação.")
    X_train, y_train = train[RAW_FEATURES], train["Survived"]
    X_test, y_test = test[RAW_FEATURES], test["Survived"]
    output_dir = Path(output_dir).resolve()
    reports, models = output_dir / "reports", output_dir / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metadata = experiment_metadata(csv_path)
    metadata.update({"train_rows": len(train), "test_rows": len(test),
                     "train_passenger_ids": train["PassengerId"].tolist(),
                     "test_passenger_ids": test["PassengerId"].tolist()})
    # SQLite guarda métricas; cada execução guarda seus próprios artefatos locais.
    mlflow.set_tracking_uri(f"sqlite:///{(output_dir / 'mlflow.db').as_posix()}")
    experiment_name = "titanic-aprendizado"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(output_dir / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    winner, searches, comparison = compare_models(X_train, y_train, cv, reports, metadata)
    pipeline = searches[winner].best_estimator_
    print(f"Modelo escolhido: {winner}. Ajustando limiar somente no treino...", flush=True)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold, thresholds = choose_threshold(y_train, oof)
    thresholds.to_csv(reports / "limiares.csv", index=False)
    # GridSearchCV já reajustou o melhor pipeline nos 80% completos do treino.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    dummy_probabilities = searches["dummy"].best_estimator_.predict_proba(X_test)[:, 1]
    final = {
        "selected_model": winner, "parameters": searches[winner].best_params_, "threshold": threshold,
        "test_note": TEST_NOTE,
        "dummy": evaluate(y_test, dummy_probabilities, 0.5),
        "default_threshold": evaluate(y_test, probabilities, 0.5),
        "selected_threshold": evaluate(y_test, probabilities, threshold),
        "confusion_matrix": confusion_matrix(y_test, probabilities >= threshold, labels=[0, 1]).tolist(),
    }
    # Inspeção posterior à avaliação: nunca alimenta a busca ou a escolha do limiar.
    permuted = permutation_importance(pipeline, X_test, y_test, scoring="f1", n_repeats=10, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": RAW_FEATURES, "importance_mean": permuted.importances_mean,
                               "importance_std": permuted.importances_std})
    importance.to_csv(reports / "importancia.csv", index=False)
    save_plots(comparison, thresholds, importance, y_test, probabilities, threshold, reports)
    chosen = final["selected_threshold"]
    top = importance.sort_values("importance_mean", ascending=False).head(3)["feature"].tolist()
    conclusions = [
        f"1. A sobrevivência no treino foi {y_train.mean():.1%}; o baseline teve acurácia {final['dummy']['accuracy']:.3f} no teste.",
        f"2. {winner} venceu pela validação cruzada com F1 médio {comparison.set_index('model').loc[winner, 'cv_f1']:.3f}; complexidade não garante melhor resultado.",
        f"3. O limiar {threshold:.2f}, escolhido somente no treino, resultou em precisão {chosen['precision']:.3f}, recall {chosen['recall']:.3f} e F1 {chosen['f1']:.3f} no teste (F1 com limiar 0,50: {final['default_threshold']['f1']:.3f}).",
        f"4. As maiores reduções médias de F1 por permutação envolveram {', '.join(top)}; são sinais preditivos conjuntos, não evidência causal ou políticas de resgate.",
        "5. O teste já foi consultado, familiares podem compartilhar padrões e há dados ausentes; resultados são didáticos e não sustentam uso operacional.",
    ]
    (reports / "conclusoes.md").write_text("\n".join(conclusions) + "\n", encoding="utf-8")
    (reports / "metricas.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    package = {"pipeline": pipeline, "threshold": threshold, "raw_features": RAW_FEATURES,
               "model": winner, "metadata": metadata}
    joblib.dump(package, models / "titanic.joblib")
    with mlflow.start_run(run_name="avaliacao-final-didatica"):
        mlflow.set_tags({"stage": "final_evaluation", "test_previously_consulted": "true"})
        mlflow.log_params({"model": winner, "threshold": threshold, "seed": SEED, **searches[winner].best_params_})
        for label in ["dummy", "default_threshold", "selected_threshold"]:
            mlflow.log_metrics({f"{label}_{key}": value for key, value in final[label].items()})
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        mlflow.log_artifact(str(models / "titanic.joblib"), artifact_path="model")
    print("\n".join(conclusions), flush=True)
    print(f"\nRelatórios: {reports}\nModelo: {models / 'titanic.joblib'}", flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description="Compara modelos, ajusta limiar e registra experimentos locais.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/train.csv")
    args = parser.parse_args()
    try:
        run_training(args.data)
    except ValueError as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
