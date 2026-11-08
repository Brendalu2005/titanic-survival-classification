"""Verificações dos pontos críticos; executa com unittest, sem dependência extra."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.pipeline import CATEGORICAL, NUMERIC, RAW_FEATURES, SEED, build_pipeline, create_features, read_passengers
from src.predict import predict_csv
from src.train import choose_threshold


def sample():
    return pd.DataFrame({
        "PassengerId": range(1, 9),
        "Name": ["A, Mr. A", "B, Mrs. B", "C, Master. C", "D, Miss. D",
                 "E, Mr. E", "F, Mme. F", "G, Dr. G", "H, Mr. H"],
        "Pclass": [3, 1, 2, 1, 3, 2, 1, 3],
        "Sex": ["male", "female", "male", "female", "male", "female", "male", "male"],
        "Age": [22, 38, np.nan, 12, 35, 28, 52, 45],
        "SibSp": [0, 1, 1, 0, 0, 0, 0, 0],
        "Parch": [0, 0, 1, 0, 0, 1, 0, 0],
        "Fare": [0, 71, 25, 80, 8, 30, 40, 7],
        "Cabin": [np.nan, "C1", np.nan, "B2", np.nan, np.nan, "A1", np.nan],
        "Embarked": ["S", "C", np.nan, "C", "S", "S", "Q", "S"],
        "Survived": [0, 1, 1, 1, 0, 1, 0, 0],
        "Desfecho": ["Não usar"] * 8,
    })


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.data = sample()
        self.pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED), scale=True)

    def test_feature_values_and_target_exclusion(self):
        features = create_features(self.data)
        self.assertEqual(features.columns.tolist(), NUMERIC + CATEGORICAL)
        self.assertEqual(features.loc[0, "LogFare"], 0)
        self.assertEqual(features.loc[2, "FamilySize"], 3)
        self.assertEqual(features.loc[5, "Title"], "Mrs")
        self.assertEqual(features.loc[6, "Title"], "Rare")
        for excluded in ["Survived", "Desfecho", "PassengerId", "Name", "Cabin", "Ticket"]:
            self.assertNotIn(excluded, features)

    def test_missing_values_and_unknown_categories(self):
        fitted = self.pipeline.fit(self.data[RAW_FEATURES], self.data["Survived"])
        new = self.data[RAW_FEATURES].iloc[:2].copy()
        new["Age"] = np.nan
        new["Cabin"] = np.nan
        new["Embarked"] = ["NOVO", np.nan]
        probability = fitted.predict_proba(new)
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_imputer_learns_only_training_partition(self):
        cv = StratifiedKFold(2, shuffle=True, random_state=SEED)
        for train_index, validation_index in cv.split(self.data, self.data["Survived"]):
            train = self.data.iloc[train_index]
            validation = self.data.iloc[validation_index][RAW_FEATURES].copy()
            validation["Age"] = 10000
            fitted = clone(self.pipeline).fit(train[RAW_FEATURES], train["Survived"])
            imputer = fitted.named_steps["preparation"].named_transformers_["numeric"].named_steps["imputer"]
            self.assertAlmostEqual(imputer.statistics_[0], train["Age"].median())
            before = imputer.statistics_.copy()
            fitted.predict_proba(validation)
            np.testing.assert_array_equal(imputer.statistics_, before)

    def test_round_trip_and_prediction_without_refit(self):
        self.pipeline.fit(self.data[RAW_FEATURES], self.data["Survived"])
        package = {"pipeline": self.pipeline, "threshold": 0.65, "raw_features": RAW_FEATURES}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.joblib"
            csv = directory / "input.csv"
            output = directory / "output.csv"
            joblib.dump(package, model)
            loaded = joblib.load(model)
            expected = self.pipeline.predict_proba(self.data[RAW_FEATURES])[:, 1]
            np.testing.assert_allclose(loaded["pipeline"].predict_proba(self.data[RAW_FEATURES])[:, 1], expected)
            self.data.drop(columns=["Survived", "Desfecho"]).to_csv(csv, index=False)
            result = predict_csv(csv, model, output)
            np.testing.assert_allclose(result["ProbabilidadeSobrevivencia"], expected)
            np.testing.assert_array_equal(result["Survived"], expected >= 0.65)
            self.assertEqual(result["PassengerId"].tolist(), self.data["PassengerId"].tolist())
            self.assertTrue(output.is_file())
            with self.assertRaisesRegex(ValueError, "diferente"):
                predict_csv(csv, model, csv)

    def test_missing_file_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with self.assertRaisesRegex(ValueError, "CSV não encontrado"):
                read_passengers(path)
            self.data.drop(columns=["Age"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Age"):
                read_passengers(path)
            with self.assertRaisesRegex(ValueError, "Modelo não encontrado"):
                predict_csv(path, Path(directory) / "absent.joblib")

    def test_invalid_inputs(self):
        for column, value in [("Fare", -1), ("Age", np.inf), ("Survived", 2)]:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.csv"
                bad = self.data.copy()
                bad.loc[0, column] = value
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    read_passengers(path, training=True)

    def test_split_and_threshold_reproducibility(self):
        first = train_test_split(self.data, stratify=self.data["Survived"], test_size=0.25, random_state=SEED)
        second = train_test_split(self.data, stratify=self.data["Survived"], test_size=0.25, random_state=SEED)
        self.assertEqual(first[0]["PassengerId"].tolist(), second[0]["PassengerId"].tolist())
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, table = choose_threshold(y, probabilities)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(choose_threshold(y, probabilities)[0], threshold)
        self.assertEqual(len(table), 13)

    def test_threshold_tie_prefers_larger_after_distance(self):
        # F1 máximo empatado apenas em 0,45 e 0,55, equidistantes de 0,50.
        from unittest.mock import patch
        with patch("src.train.evaluate", side_effect=lambda y, p, t: {
            "f1": 1.0 if round(float(t), 2) in [0.45, 0.55] else 0.0,
        }):
            threshold, _ = choose_threshold([0, 1], [0.1, 0.9])
        self.assertEqual(threshold, 0.55)


if __name__ == "__main__":
    unittest.main()
