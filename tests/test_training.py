import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml.features import EXCLUDED_FEATURES, NUMERIC_FEATURES


def test_artifacts_written(trained):
    for name in ("model.joblib", "metrics.json", "model_info.json"):
        assert (trained["artifact_dir"] / name).exists()


def test_both_models_evaluated_with_all_metrics(trained):
    models = trained["metrics"]["models"]
    assert set(models) == {"logistic_regression", "random_forest"}
    for result in models.values():
        test = result["test"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= test[key] <= 1.0
        cm = test["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(trained["X_test"])


def test_selected_model_has_lowest_cv_cost(trained):
    models = trained["metrics"]["models"]
    selected = trained["metrics"]["selected_model"]
    assert selected == min(models, key=lambda n: models[n]["cv"]["mean_cost_per_applicant"])


def test_saved_metrics_match_predictions_of_reloaded_model(trained):
    """Metrics on disk must be reproducible from the reloaded artifact's real predictions."""
    bundle = joblib.load(trained["artifact_dir"] / "model.joblib")
    saved = json.loads((trained["artifact_dir"] / "metrics.json").read_text())
    selected = saved["models"][saved["selected_model"]]["test"]

    proba = bundle["pipeline"].predict_proba(trained["X_test"])[:, 1]
    y = trained["y_test"].to_numpy()
    pred = (proba >= bundle["threshold"]).astype(int)

    assert roc_auc_score(y, proba) == pytest.approx(selected["roc_auc"])
    assert (pred == y).mean() == pytest.approx(selected["accuracy"])
    assert int(((pred == 1) & (y == 1)).sum()) == selected["confusion_matrix"]["tp"]


def test_preprocessing_fitted_on_training_split_only(trained):
    scaler = trained["bundle"]["pipeline"].named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    train_means = trained["X_train"][NUMERIC_FEATURES].astype(float).mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_means)
    assert scaler.n_samples_seen_ == len(trained["X_train"])


def test_pipeline_imputes_missing_values(trained):
    row = trained["X_test"].iloc[[0]].astype({f: float for f in NUMERIC_FEATURES})
    row.loc[:, ["credit_amount", "age_years"]] = np.nan
    row.loc[:, "savings_status"] = None
    proba = trained["bundle"]["pipeline"].predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_sensitive_attributes_not_used(trained):
    for feature in EXCLUDED_FEATURES:
        assert feature not in trained["bundle"]["features"]
