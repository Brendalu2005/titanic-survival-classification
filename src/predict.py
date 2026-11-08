"""Previsão em lote: python -m src.predict --input data/test.csv."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import RAW_FEATURES, ROOT, read_passengers


def predict_csv(input_path, model_path=ROOT / "models/titanic.joblib", output_path=ROOT / "predictions/previsoes.csv"):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ValueError(f"Modelo não encontrado: {model_path}. Execute python -m src.train primeiro.")
    # Carregue somente o artefato gerado por este projeto no seu computador.
    package = joblib.load(model_path)
    data = read_passengers(input_path)
    if package["raw_features"] != RAW_FEATURES:
        raise ValueError("As colunas do modelo salvo diferem das esperadas por esta versão do projeto.")
    probability = package["pipeline"].predict_proba(data[RAW_FEATURES])[:, 1]
    result = pd.DataFrame({
        "PassengerId": data["PassengerId"],
        "ProbabilidadeSobrevivencia": probability,
        "Survived": (probability >= package["threshold"]).astype(int),
    })
    if not np.isfinite(probability).all():
        raise ValueError("O modelo produziu probabilidades inválidas.")
    output_path = Path(output_path)
    if output_path.resolve() in {Path(input_path).resolve(), model_path.resolve()}:
        raise ValueError("O arquivo de saída deve ser diferente dos dados e do modelo.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Gera previsões didáticas para um CSV de passageiros.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/titanic.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions/previsoes.csv")
    args = parser.parse_args()
    try:
        result = predict_csv(args.input, args.model, args.output)
    except ValueError as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"{len(result)} previsões salvas em {args.output}")


if __name__ == "__main__":
    main()
