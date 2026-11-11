"""O baseline não deve aprender nem selecionar limiares com os 20% de teste."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.baseline import run_baseline
from src.pipeline import SEED
from tests.test_workflow import sample


class BaselineTests(unittest.TestCase):
    def test_holdout_features_do_not_change_baseline_or_threshold(self):
        data = pd.concat([sample()] * 5, ignore_index=True)
        data["PassengerId"] = range(1, len(data) + 1)
        _, holdout = train_test_split(
            data, test_size=0.2, stratify=data["Survived"], random_state=SEED,
        )
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "train.csv"
            data.to_csv(csv, index=False)
            original = run_baseline(csv)
            # Valores extremos e categorias inéditas só no teste reservado.
            data.loc[holdout.index, ["Age", "Fare"]] = 1000000
            data.loc[holdout.index, "Embarked"] = "TESTE_NAO_USAR"
            data.loc[holdout.index, "Name"] = "Teste, Dr. Ignorar"
            data.to_csv(csv, index=False)
            changed = run_baseline(csv)
            pd.testing.assert_frame_equal(original["comparison"], changed["comparison"])
            pd.testing.assert_frame_equal(original["thresholds"], changed["thresholds"])
            self.assertEqual(original["threshold"], changed["threshold"])
            self.assertEqual(original["train_rows"], 32)
            self.assertEqual(list(Path(directory).iterdir()), [csv])


if __name__ == "__main__":
    unittest.main()
