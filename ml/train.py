"""Reproducible training + evaluation for the credit-risk models.

Usage (from the project root):
    .venv/bin/python -m ml.train

Procedure
1. Load the German Credit dataset and split 80/20 (stratified) BEFORE any
   preprocessing is fitted, so the test set never influences transforms.
2. Each candidate is a single sklearn Pipeline (impute -> scale/one-hot ->
   classifier), so preprocessing is re-fitted inside every CV fold too.
3. Model + decision threshold selection uses only the training set: 5-fold
   stratified out-of-fold probabilities, scored with the dataset's cost matrix
   (FN = 5, FP = 1). Lowest mean cost wins; ROC-AUC breaks ties.
4. The test set is used once, to report final held-out metrics.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data import DEFAULT_DATA_PATH, PROJECT_ROOT, load_dataset
from ml.features import (
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    category_options,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_candidates():
    # class_weight="balanced" counters the 70/30 good/bad imbalance.
    return {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(
                class_weight="balanced", C=1.0, max_iter=2000, random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def misclassification_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp)


def best_threshold(y_true, proba):
    costs = [misclassification_cost(y_true, (proba >= t).astype(int)) for t in THRESHOLD_GRID]
    i = int(np.argmin(costs))
    return float(THRESHOLD_GRID[i]), costs[i] / len(y_true)


def evaluate(y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, proba)
    cost = misclassification_cost(y_true, y_pred)
    observed, predicted = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "total_cost": cost,
        "mean_cost_per_applicant": cost / len(y_true),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": [
            {"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)
        ],
        # Calibration diagnostics: class_weight="balanced" shifts scores upward, so the
        # outputs are risk scores, not calibrated probabilities.
        "calibration": {
            "brier_score": brier_score_loss(y_true, proba),
            "mean_predicted_bad": float(proba.mean()),
            "observed_bad_rate": float(y_true.mean()),
            "reliability_bins": [
                {"mean_predicted": float(p), "observed_rate": float(o)} for p, o in zip(predicted, observed)
            ],
        },
        "n_samples": int(len(y_true)),
    }


def top_features(pipeline, n=10):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = model.coef_[0]
        kind = "coefficient (log-odds of bad risk, standardised inputs)"
    else:
        values = model.feature_importances_
        kind = "impurity-based importance"
    order = np.argsort(-np.abs(values))[:n]
    return kind, [{"feature": str(names[i]), "value": float(values[i])} for i in order]


def train(data_path=None, artifact_dir=None, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    artifact_dir = Path(artifact_dir or os.environ.get("CREDIT_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    log(f"Loaded {len(df)} rows; bad-risk rate {df[TARGET].mean():.3f}")

    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    log(f"Train {len(X_train)} / Test {len(X_test)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, pipeline in build_candidates().items():
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, cv_cost = best_threshold(y_train.to_numpy(), oof)
        cv_auc = roc_auc_score(y_train, oof)

        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        importance_kind, importances = top_features(pipeline)

        results[name] = {
            "cv": {"folds": CV_FOLDS, "mean_cost_per_applicant": cv_cost,
                   "roc_auc": cv_auc, "selected_threshold": threshold},
            "test": evaluate(y_test.to_numpy(), test_proba, threshold),
            "test_at_0_5": evaluate(y_test.to_numpy(), test_proba, 0.5),
            "importance_kind": importance_kind,
            "top_features": importances,
        }
        fitted[name] = pipeline
        t = results[name]["test"]
        log(f"[{name}] CV cost {cv_cost:.3f} AUC {cv_auc:.3f} thr {threshold:.2f} | "
            f"test acc {t['accuracy']:.3f} prec {t['precision']:.3f} rec {t['recall']:.3f} "
            f"f1 {t['f1']:.3f} auc {t['roc_auc']:.3f} cost {t['total_cost']}")

    selected = min(results, key=lambda n: (results[n]["cv"]["mean_cost_per_applicant"],
                                          -results[n]["cv"]["roc_auc"]))
    log(f"Selected: {selected}")

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_info = {
        "selected_model": selected,
        "algorithm": type(fitted[selected].named_steps["model"]).__name__,
        "decision_threshold": results[selected]["cv"]["selected_threshold"],
        "selection_criterion": (
            f"Lowest mean misclassification cost on {CV_FOLDS}-fold stratified out-of-fold "
            f"predictions of the training set, using the dataset's cost matrix "
            f"(false negative = {COST_FALSE_NEGATIVE}, false positive = {COST_FALSE_POSITIVE}). "
            "Decision threshold tuned the same way. Ties broken by CV ROC-AUC. "
            "The test set was not used for selection."
        ),
        "positive_class": CLASS_LABELS[1],
        "class_labels": CLASS_LABELS,
        "label_mapping": "Raw label 1 (good) -> 0 good_credit_risk; raw label 2 (bad) -> 1 bad_credit_risk",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": category_options(),
        "excluded_features": EXCLUDED_FEATURES,
        # Training-set median (numeric) / mode (categorical): used to prefill the UI form.
        "feature_defaults": {
            **{c: float(X_train[c].median()) for c in NUMERIC_FEATURES},
            **{c: str(X_train[c].mode()[0]) for c in CATEGORICAL_FEATURES},
        },
        "dataset": {
            "name": "Statlog (German Credit Data)",
            "source": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "license": "CC BY 4.0",
            "path": str(Path(data_path or DEFAULT_DATA_PATH).relative_to(PROJECT_ROOT))
            if Path(data_path or DEFAULT_DATA_PATH).is_relative_to(PROJECT_ROOT) else str(data_path),
            "n_rows": int(len(df)),
            "bad_rate": float(df[TARGET].mean()),
        },
        "split": {"test_size": TEST_SIZE, "stratified": True, "random_state": RANDOM_STATE,
                  "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "class_imbalance_handling": "class_weight='balanced' + cost-based threshold tuning",
        "probability_calibration": (
            "Not calibrated. class_weight='balanced' re-weights the minority class, so predicted "
            "P(bad) overstates the observed bad rate (see metrics.json -> test.calibration). Treat "
            "the output as a risk score for ranking and thresholding, not as a literal default probability."
        ),
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
    }
    metrics = {"selected_model": selected, "trained_at": trained_at, "models": results}

    # The pipeline object holds both the fitted preprocessing and the classifier.
    joblib.dump({
        "pipeline": fitted[selected],
        "threshold": model_info["decision_threshold"],
        "features": MODEL_FEATURES,
        "model_name": selected,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
    }, artifact_dir / "model.joblib")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifact_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))
    log(f"Saved artifacts to {artifact_dir}")
    return {"metrics": metrics, "model_info": model_info,
            "X_test": X_test, "y_test": y_test, "X_train": X_train}


if __name__ == "__main__":
    train()
