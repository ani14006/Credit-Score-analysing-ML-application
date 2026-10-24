"""Instrument a real training run and prove the test split never reaches fitting or selection."""

import pytest
from sklearn.pipeline import Pipeline

import ml.train as train_module


@pytest.fixture(scope="module")
def instrumented_run(tmp_path_factory):
    fit_indices, cv_indices, threshold_sizes = [], [], []
    mp = pytest.MonkeyPatch()

    original_fit = Pipeline.fit
    def recording_fit(self, X, y=None, **kw):
        fit_indices.append(set(X.index))
        return original_fit(self, X, y, **kw)
    mp.setattr(Pipeline, "fit", recording_fit)

    original_cvp = train_module.cross_val_predict
    def recording_cvp(estimator, X, y, **kw):
        cv_indices.append(set(X.index))
        return original_cvp(estimator, X, y, **kw)
    mp.setattr(train_module, "cross_val_predict", recording_cvp)

    original_bt = train_module.best_threshold
    def recording_bt(y_true, proba):
        threshold_sizes.append(len(y_true))
        return original_bt(y_true, proba)
    mp.setattr(train_module, "best_threshold", recording_bt)

    try:
        result = train_module.train(artifact_dir=tmp_path_factory.mktemp("leak"), verbose=False)
    finally:
        mp.undo()
    return result, fit_indices, cv_indices, threshold_sizes


def test_split_is_disjoint(instrumented_run):
    result = instrumented_run[0]
    assert not set(result["X_train"].index) & set(result["X_test"].index)
    assert len(result["X_train"]) + len(result["X_test"]) == 1000


def test_no_fit_ever_sees_test_rows(instrumented_run):
    result, fit_indices, _, _ = instrumented_run
    train_idx, test_idx = set(result["X_train"].index), set(result["X_test"].index)
    # 2 models x (5 CV folds + 1 final refit)
    assert len(fit_indices) == 12
    for idx in fit_indices:
        assert idx <= train_idx
        assert not idx & test_idx


def test_cross_validation_and_threshold_use_training_data_only(instrumented_run):
    result, _, cv_indices, threshold_sizes = instrumented_run
    train_idx = set(result["X_train"].index)
    assert cv_indices == [train_idx, train_idx]
    assert threshold_sizes == [len(train_idx), len(train_idx)]


def test_selection_uses_cv_results_not_test_results(instrumented_run):
    metrics = instrumented_run[0]["metrics"]
    models = metrics["models"]
    by_cv = min(models, key=lambda n: models[n]["cv"]["mean_cost_per_applicant"])
    assert metrics["selected_model"] == by_cv
    for result in models.values():
        assert result["test"]["threshold"] == result["cv"]["selected_threshold"]
