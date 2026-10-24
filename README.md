# Credit Risk Scoring

An end-to-end credit-risk classifier: a scikit-learn pipeline trained on the public
**UCI Statlog (German Credit Data)** dataset, served by a FastAPI backend, with a React + Vite
dashboard. Every number in the UI comes from the backend, which reads it from artifacts
written by the training script.

> **Scope.** The model estimates the probability that an applicant falls in the dataset's
> "bad credit risk" class. That is a default-risk classifier trained on 1,000 applications
> from 1994 Germany. It is not a complete real-world credit score and should not be the
> only basis for a lending decision.

## Project layout

```
credit-scoring/
├── data/
│   ├── README.md            dataset source, license, label mapping, excluded features
│   └── raw/german.data      UCI dataset (unmodified) + german.doc
├── ml/
│   ├── features.py          single feature schema shared by training and API
│   ├── data.py              loading + code decoding
│   └── train.py             reproducible training / evaluation / selection
├── artifacts/               output of ml/train.py
│   ├── model.joblib         fitted Pipeline (preprocessing + selected classifier) + threshold
│   ├── metrics.json         test-set and CV metrics for both models, ROC points, confusion matrices
│   └── model_info.json      selected algorithm, selection criterion, schema, dataset info
├── backend/app/
│   ├── main.py              FastAPI app: /health /metrics /model-info /predict /predictions
│   ├── schemas.py           request validation (Pydantic), generated from ml/features.py
│   └── store.py             artifact loading, inference, SQLite prediction log
├── frontend/                React + Vite + Recharts dashboard
├── tests/                   pytest suite (data, training, API)
├── requirements.txt
└── .env.example
```

## Commands

Requirements: Python 3.12 and Node 18 or newer. Run everything from the project root.

```bash
# 1. Install
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
(cd frontend && npm install)

# 2. Train (about 10 s; writes artifacts/)
.venv/bin/python -m ml.train

# 3. Start the backend → http://localhost:8000  (interactive docs at /docs)
.venv/bin/uvicorn backend.app.main:app --port 8000

# 4. Start the frontend (second terminal) → http://localhost:5173
cd frontend && npm run dev

# 5. Run the tests
.venv/bin/python -m pytest -v
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `CREDIT_ARTIFACT_DIR` | `artifacts` | where training writes and the API reads the model |
| `CREDIT_PREDICTIONS_DB` | `var/predictions.db` | SQLite file for prediction history |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated allowed frontend origins |
| `CREDIT_DATA_PATH` | `data/raw/german.data` | dataset file used by training |
| `VITE_API_URL` (frontend) | `http://localhost:8000` | backend base URL; see `frontend/.env.example` |

## Machine-learning pipeline

1. **Data.** 1,000 applicants, 20 attributes, 700 good and 300 bad. The source and license
   (CC BY 4.0) are in [data/README.md](data/README.md). Symbolic codes are decoded into
   readable categories.
2. **Target.** Raw `1` (good) → `0 good_credit_risk`; raw `2` (bad) → `1 bad_credit_risk`,
   the positive class.
3. **Features.** 7 numeric and 11 categorical. `personal_status_sex` (encodes sex) and
   `foreign_worker` (national origin) are excluded from the model.
4. **Split first.** A stratified 80/20 train/test split (`random_state=42`) is made before
   anything is fitted.
5. **Preprocessing** runs inside one sklearn `Pipeline`, so it is fitted only on training
   data, and refitted within each CV fold:
   - numeric: median imputation → `StandardScaler`
   - categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

   The dataset has no missing values, but the imputers are in place and tested.
6. **Class imbalance.** Both models use `class_weight="balanced"`, and the decision
   threshold is tuned to the cost matrix.
7. **Models.**
   - `LogisticRegression(C=1.0)`
   - `RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt")`
8. **Selection criterion.** The dataset ships a cost matrix: a bad applicant classified as
   good costs 5, a good applicant classified as bad costs 1. For each model:
   - compute 5-fold stratified out-of-fold probabilities on the training set;
   - choose the threshold (grid 0.05–0.95) that minimises mean cost;
   - pick the model with the lowest CV cost, breaking ties on CV ROC-AUC.

   The test set is not used for any of this.
9. **Evaluation.** Each refitted model is scored once on the 200 held-out applicants:
   accuracy, precision, recall, F1, ROC-AUC, confusion matrix, total cost and ROC curve.
   Metrics at the default 0.5 threshold are also saved (`test_at_0_5` in `metrics.json`).
   Calibration diagnostics are saved too (`test.calibration`): Brier score and reliability bins.
10. **Persistence.** `joblib` saves the fitted pipeline, which contains both the
    preprocessing and the classifier, together with the threshold and feature list.

### Results (from `artifacts/metrics.json`; test n = 200)

| Model | CV cost/applicant | CV ROC-AUC | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Test cost | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.515 | 0.759 | 0.28 | 0.565 | 0.398 | 0.883 | 0.549 | 0.810 | 115 | 60/80/7/53 |
| **Random Forest (selected)** | **0.504** | **0.795** | 0.36 | 0.635 | 0.443 | 0.850 | 0.583 | 0.792 | **109** | 76/64/9/51 |

How to read these numbers:
- **Why Random Forest.** It had the lower CV cost, so it was selected. It also had the lower
  cost on the test set (109 vs 115).
- **Logistic Regression ranks better on the test set** (ROC-AUC 0.810 vs 0.792). The gap is
  small for n = 200, and selection deliberately does not look at test results.
- **Thresholds are low (0.28 / 0.36) on purpose.** Missing a bad applicant costs 5× more, so
  both models flag aggressively. That gives high recall on bad risks and modest precision.
- **Accuracy is low.** It is not the optimisation target. Predicting "good" for everyone
  would score 70% accuracy but cost 300.

### Limitation: probabilities are not calibrated

Both models train with `class_weight="balanced"`, which up-weights the 30% bad class. As a
result, `probability_bad` overstates risk. On the test set the Random Forest's mean predicted
P(bad) is about 0.41, while the observed bad rate is 0.30. Applicants scored about 0.29 were
bad about 12% of the time (see `test.calibration.reliability_bins`).

The cost-optimal threshold (0.36) was tuned on these same uncalibrated scores, so the
**classification** is consistent. Treat the number as a risk score for ranking and
thresholding, not as a literal default probability. Calibrated probabilities would need a
calibration step fitted inside the training data, for example `CalibratedClassifierCV`, and
are not implemented.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{status, model_loaded, model_name, error}`, always 200 |
| GET | `/metrics` | contents of `metrics.json`; 503 if artifacts are missing |
| GET | `/model-info` | contents of `model_info.json`; 503 if artifacts are missing |
| POST | `/predict` | validates the applicant, runs the saved pipeline, logs and returns the prediction |
| GET | `/predictions?limit=20` | prediction history persisted in SQLite |

Example:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "duration_months": 24, "credit_amount": 4000, "installment_rate_pct": 3, "residence_since": 2,
  "age_years": 35, "existing_credits": 1, "num_dependents": 1,
  "checking_status": "lt_0_dm", "credit_history": "existing_paid_duly", "purpose": "car_new",
  "savings_status": "lt_100_dm", "employment_since": "1_to_4_years", "other_debtors": "none",
  "property": "car_or_other", "other_installment_plans": "none", "housing": "own",
  "job": "skilled_employee", "telephone": "none"}'
```

The response includes `prediction`, `predicted_class`, `probability_bad`,
`probability_good`, `decision_threshold`, `model_name`, `warnings` and `prediction_id`.

Validation rules:
- All 18 fields are required, and unknown fields are rejected. Sending
  `personal_status_sex`, for example, returns 422.
- Numeric bounds are enforced, and categories must be one of the documented values.
- Invalid input returns 422 with per-field messages.
- Values inside the bounds but outside the range seen in training are accepted, but flagged
  in `warnings`.
- The allowed values are listed in `/model-info` (`category_options`) and `/openapi.json`.
  The frontend builds its form from these.

## Frontend

- **Metrics overview** for the selected model.
- **Applicant form**:
  - fields and validation bounds are read from the backend's OpenAPI schema;
  - prefilled with training-set medians and modes;
  - server-side 422 errors are shown on the matching field.
- **Prediction result**: class, probability, and a meter marking the threshold.
- **Persisted history** read from `/predictions`.
- **Model comparison**: table, grouped metric bars, ROC curves and confusion matrices, all
  rendered from `/metrics`.
- **Light and dark mode** following the OS setting, and a responsive layout.

## Tests

`tests/` (39 tests) trains from the real dataset into a temporary directory and checks:

- **Data loading**: dataset loads with 1,000 rows and a 700/300 label split; all codes decode.
- **Saved metrics are real**: metrics on disk equal metrics recomputed from the reloaded
  `model.joblib`'s own predictions.
- **No leakage**: the fitted scaler's means equal the training-split means, and it saw
  exactly 800 rows.
- **Missing values**: the imputers handle NaN input.
- **Leakage, checked during a real training run** (`test_leakage.py`): every `Pipeline.fit`
  call (12 in total) sees only training rows, and cross-validation and threshold tuning
  receive only the 800 training rows. Selection follows the CV results. These tests fail if
  CV or threshold tuning is pointed at the full dataset or the test set.
- **Model loading** (`test_model_store.py`): the real artifact loads, and `predict` calls the
  pipeline's `predict_proba` exactly once. A corrupted or feature-mismatched artifact is
  reported, and `/predict` then returns 503. Calibration diagnostics are present.
- **Protected attributes** are absent from the model.
- **`/predict` uses the model**: its `probability_bad` equals `pipeline.predict_proba`
  for the same row. Across 25 test rows the API matches the model and returns more than 10
  distinct values, so it is not a canned answer. A riskier profile scores higher.
- **Validation**: invalid, missing and unknown fields return 422.
- **Missing artifacts**: `/health` reports the problem, and the model endpoints return 503.
- **Endpoint contents**: `/metrics` and `/model-info` equal the artifact files.
- **Bad requests**: an empty body, wrong types, non-JSON input and invalid `limit` values
  all return 422.
- **History**: predictions persist in SQLite and are still there after the app restarts,
  checked through the API and by querying the database directly.
