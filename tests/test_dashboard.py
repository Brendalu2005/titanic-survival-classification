"""Testes do painel: métricas permanecem ligadas à avaliação, não aos filtros."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard_data import ROOT, load_evaluation, load_predictions

APP = ROOT / "streamlit_app.py"


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "reports").mkdir()
        (self.root / "predictions").mkdir()
        (self.root / "data").mkdir()
        scores = {"accuracy": 0.85, "precision": 9 / 11, "recall": 0.9, "f1": 0.857}
        metrics = {
            "selected_model": "logistica", "threshold": 0.45, "parameters": {"model__C": 10},
            "selected_threshold": scores, "default_threshold": scores,
            "dummy": {"accuracy": 0.5, "precision": 0, "recall": 0, "f1": 0},
            "confusion_matrix": [[8, 2], [1, 9]],
            "test_note": "Avaliação didática em teste previamente consultado.",
        }
        (self.root / "reports/metricas.json").write_text(json.dumps(metrics), encoding="utf-8")
        pd.DataFrame({
            "PassengerId": [892, 893, 894], "ProbabilidadeSobrevivencia": [0.1, 0.8, 0.5], "Survived": [0, 1, 1],
        }).to_csv(self.root / "predictions/previsoes.csv", index=False)
        pd.DataFrame({
            "PassengerId": [892, 893, 894], "Name": ["Kelly", "Ada", "Maria"],
            "Age": [30, 20, 40], "Pclass": [3, 1, 2], "Sex": ["male", "female", "female"],
        }).to_csv(self.root / "data/test.csv", index=False)
        self.root_patch = patch("src.dashboard_data.ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.directory.cleanup()

    def app(self):
        return AppTest.from_file(str(APP), default_timeout=30).run()

    def test_filters_do_not_change_evaluation_metrics(self):
        app = self.app()
        self.assertFalse(app.exception)
        initial = [metric.value for metric in app.metric]
        self.assertEqual(initial, ["81,8%", "90,0%", "85,0%"])
        self.assertEqual(len(app.dataframe[0].value), 3)
        app.text_input(key="search").set_value("Ada").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 1)
        self.assertEqual(app.dataframe[0].value.iloc[0]["ID"], 893)
        self.assertEqual([metric.value for metric in app.metric], initial)
        app.text_input(key="search").set_value("").run()
        app.slider(key="probability").set_value((60, 100)).run()
        self.assertEqual(app.dataframe[0].value["ID"].tolist(), [893])

    def test_empty_filter_and_evaluation_view(self):
        app = self.app()
        app.text_input(key="search").set_value("inexistente").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)
        self.assertTrue(any("Nenhum passageiro" in item.value for item in app.info))
        app.segmented_control(key="view").set_value("Avaliação").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.shape, (3, 4))
        self.assertEqual(app.table[0].value.to_numpy().tolist(), [[8, 2], [1, 9]])
        self.assertTrue(any("previamente consultado" in item.value for item in app.info))

    def test_missing_outputs_have_instructions(self):
        (self.root / "reports/metricas.json").unlink()
        (self.root / "predictions/previsoes.csv").unlink()
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("python -m src.train" in item.value for item in app.info))
        self.assertTrue(any("python -m src.predict" in item.value for item in app.info))

    def test_optional_passenger_context_and_invalid_csv(self):
        path = self.root / "predictions/previsoes.csv"
        data = load_predictions(path, self.root / "data/absent.csv")
        self.assertEqual(len(data), 3)
        self.assertNotIn("Name", data)
        data.loc[0, "ProbabilidadeSobrevivencia"] = 1.5
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "inválid"):
            load_predictions(path)

    def test_invalid_report_is_not_shown_as_metrics(self):
        path = self.root / "reports/metricas.json"
        path.write_text('{"selected_threshold": {"accuracy": 42}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "avaliação"):
            load_evaluation(path.parent)
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe[0].value), 3)


if __name__ == "__main__":
    unittest.main()
