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
│   ├── main.py              FastAPI app: /health /metrics /model-info /predict /predictions[/lookup]
│   ├── schemas.py           request validation (Pydantic), generated from ml/features.py
│   ├── store.py             artifact loading, inference, SQLite prediction log, access tokens
│   ├── limits.py            request body size limit
│   └── agent/               AI Financial Wellness Agent
│       ├── routes.py        /agent/* endpoints (token-checked)
│       ├── service.py       verified context -> LLM -> validated output -> SQLite
│       ├── context.py       context builder, limitations, optional user-supplied details
│       ├── explainability.py model sensitivity (what-if) analysis on the saved pipeline
│       ├── prompts.py       fixed system prompt, output JSON schemas and validators
│       ├── llm.py           Anthropic Claude provider (server-side key, timeouts, error mapping)
│       ├── storage.py       sessions, conversation turns, plans, tasks
│       └── ratelimit.py     per-client limit on LLM-calling requests
├── frontend/                React + Vite + Recharts dashboard (src/components/AgentPanel.jsx)
├── tests/                   pytest suite (data, training, leakage, API, security, agent)
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
| `CREDIT_MAX_BODY_BYTES` | `16384` | request bodies above this size get HTTP 413 |
| `CREDIT_STORE_HISTORY` | `true` | `false` stores no prediction history |
| `CREDIT_HISTORY_RETENTION_DAYS` | `30` | history rows older than this are purged (`0` keeps them forever) |
| `CREDIT_AGENT_PROVIDER` | *(auto)* | `gemini` or `anthropic`; unset = whichever key is present (Anthropic first) |
| `GEMINI_API_KEY` | *(unset)* | Google Gemini key; backend only (environment or git-ignored `.env`) |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model used by the agent |
| `ANTHROPIC_API_KEY` | *(unset)* | Anthropic key; backend only |
| `CREDIT_AGENT_MODEL` | `claude-opus-5-5` | Claude model used when the provider is `anthropic` |
| `CREDIT_AGENT_TIMEOUT_SECONDS` | `90` | per-request LLM timeout (the SDK retries twice) |
| `CREDIT_AGENT_RATE_LIMIT_PER_MINUTE` | `10` | LLM-calling agent requests per client IP per minute |
| `CREDIT_ADMIN_TOKEN` | *(unset)* | operator bearer token for `GET /predictions`; unset = listing disabled |
| `VITE_API_URL` (frontend) | `http://localhost:8000` | backend base URL; see `frontend/.env.example` |

See [SECURITY.md](SECURITY.md) for the security model, data-retention policy and the controls
required before any public deployment. There are no user accounts: each stored prediction (and
its agent session) can only be read with the access token returned when it was made.

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
| POST | `/predict` | validates the applicant, runs the saved pipeline, stores and returns the prediction plus a one-time `access_token` |
| POST | `/predictions/lookup` | `{"items": [{"id", "token"}]}` → the caller's own stored predictions (wrong tokens are skipped) |
| GET | `/predictions?limit=20` | operator listing of all predictions; needs `Authorization: Bearer $CREDIT_ADMIN_TOKEN` (403 when unset) |
| GET | `/agent/status` | whether the LLM is configured, and which model |
| POST | `/agent/text` | **simple text API**: `{prediction_id, question?}` + `X-Prediction-Token` → `text/plain` answer (no session, nothing stored) |
| POST | `/agent/sessions` | `{prediction_id, user_context?}` + `X-Prediction-Token` → AI explanation of that stored prediction |
| GET | `/agent/sessions/{id}` | explanation, evidence, conversation and latest plan |
| POST | `/agent/sessions/{id}/messages` | `{question}` → follow-up answer using the session's context |
| POST | `/agent/sessions/{id}/plan` | `{horizon_days: 30\|60\|90, goals?}` → personalised plan, saved with tasks |
| GET | `/agent/sessions/{id}/plan` | latest saved plan with progress |
| PATCH | `/agent/sessions/{id}/plan/tasks/{task_id}` | `{done}` → mark a task complete or not |

Simple text API example (use the `prediction_id` and `access_token` returned by `/predict`):

```bash
curl -X POST localhost:8000/agent/text -H 'Content-Type: application/json' \
  -H "X-Prediction-Token: $TOKEN" -d '{"prediction_id": 1, "question": "What should I focus on first?"}'
```

Leave out `question` to get a plain-text explanation of the prediction. Errors come back as plain
text with the usual status codes (404, 429, 503, 504).

All `/agent/*` session routes require the `X-Prediction-Token` header of the session's prediction;
an unknown session and a wrong token both return 404.

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
`probability_good`, `decision_threshold`, `model_name`, `warnings`, `prediction_id` and
`access_token`. Only a SHA-256 hash of the token is stored.

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
- **Your prediction history**: predictions made in this browser, read through
  `/predictions/lookup` with the tokens kept in this browser's local storage. **Open** reloads a
  past prediction and its agent session.
- **AI Financial Wellness Agent**: explanation, evidence, improvement plan and chat (below).
- **Model comparison**: table, grouped metric bars, ROC curves and confusion matrices, all
  rendered from `/metrics`.
- **Light and dark mode** following the OS setting, and a responsive layout.

## AI Financial Wellness Agent

An LLM (Google Gemini `gemini-3.5-flash`, or Anthropic Claude `claude-opus-5-5`, chosen by
`CREDIT_AGENT_PROVIDER`) explains a stored prediction, builds an
improvement plan and answers follow-up questions. It is connected to the real model and database;
it never changes the prediction.

```
Browser ── POST /agent/sessions {prediction_id} + X-Prediction-Token ──▶ FastAPI
                                                   │ load prediction row from SQLite (token checked)
                                                   │ build verified context:
                                                   │   stored label / score / threshold, submitted features,
                                                   │   sensitivity analysis (re-runs the saved pipeline),
                                                   │   test metrics, limitations, what is NOT collected,
                                                   │   user-supplied details (marked unverified)
                                                   ▼
                     Gemini or Claude API via the official SDK (key stays on the server)
                                                   │ JSON-schema output, validated with Pydantic
                                                   ▼
                     SQLite: agent_sessions / agent_messages / agent_plans / agent_plan_tasks
```

**What the agent produces.**
- **Explanation:** what the prediction means, factors to review, positive habits, risks and
  missing information, prioritised steps, an action plan, and the uncertainty and limitations.
  Every point is tagged *observed*, *model analysis*, *you reported*, *general guidance* or
  *assumption*.
- **Plan:** a 30-, 60- or 90-day plan of tasks. Each task has a week, a target date, a priority
  and why it matters, plus a checkbox; progress is saved.
- **Follow-up chat:** each answer sees the same verified context, the explanation, the earlier
  questions and the current plan.

**Feature-level explainability.** The Random Forest has no per-prediction coefficients, so
`explainability.py` re-runs the saved pipeline with one input changed at a time.
- It is labelled as model sensitivity, not causation.
- It only tests financially sensible changes: smaller or shorter loans, fewer credits, higher
  balances, no other instalment plans.
- In the 1994 data, "no checking account" and "unknown savings" look low-risk. These are never
  suggested, and both the agent and the UI say so.

**Safeguards.**
- **Prediction values come from the server.** The agent reads label, score and threshold from
  the database. A request that tries to supply its own probability is rejected (422).
- **Fixed system prompt** covering:
  - no invented salary, balances, limits or bureau scores, and no CIBIL/FICO equivalence;
  - an uncalibrated-score warning;
  - no borrowing-more advice and no promised outcomes;
  - education, not professional advice;
  - user text treated as data, not instructions.
- **No tools.** The LLM can't run commands, query the database or touch model files.
- **Failures are never papered over.** A timeout, provider error, refusal, cut-off or invalid
  JSON returns an error (502/503/504) and nothing is stored; no fallback text is fabricated. A
  missing `ANTHROPIC_API_KEY` returns a clear 503, and the UI shows a configuration error.
- **Limits:**
  - 90-second timeout per LLM request;
  - 10 LLM requests per minute per client IP;
  - 40 questions per session;
  - 16 KiB request bodies.
- **Privacy:**
  - the key is never sent to the browser (tested against the built bundle);
  - sessions are token-scoped and unguessable;
  - agent data is deleted together with its prediction by the 30-day retention purge;
  - optional details are sent to Anthropic only when you choose to add them.

**Enable it.**
```bash
cp .env.example .env               # .env is git-ignored
# edit .env: CREDIT_AGENT_PROVIDER=gemini and GEMINI_API_KEY=...  (or anthropic + ANTHROPIC_API_KEY)
.venv/bin/uvicorn backend.app.main:app --port 8000
```

## Tests

`tests/` (108 tests: 107 run offline, plus 1 opt-in live LLM test) trains from the real dataset into a
temporary directory and checks:

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
- **Security** (`test_security.py`): errors don't disclose server paths, oversized and chunked
  bodies get 413, CORS rejects unknown origins, history retention purges old rows, history can be
  disabled, and SQL-like input in `limit` is rejected.
- **History**: predictions persist in SQLite and are still there after the app restarts,
  checked through the API and by querying the database directly.

- **AI agent** (`test_agent.py`). These tests use a **mocked LLM**: a recording test double that
  shows what the backend sends. They cover:
  - the agent receives the real model prediction (checked against `predict_proba`);
  - the provider is invoked with the fixed prompt and schema;
  - user-supplied details are passed and labelled unverified;
  - follow-ups replay the same context and earlier turns;
  - a missing key returns 503; provider errors (timeout, connection, auth, rate limit, 5xx,
    refusal, cut-off, bad JSON) return errors and store nothing;
  - the stored prediction is byte-identical after the agent runs;
  - plans differ with each user's context, and their tasks save, update and survive a restart;
  - another prediction's token gets 404; rate limit 429; cascade deletion on purge.
- **Credentials** (`test_frontend_secrets.py`): the frontend source never references LLM
  credentials, and a production build made with a decoy key in the environment doesn't contain it.
- **Gemini provider** (`test_agent.py`, mocked SDK): request shape, role mapping, JSON schema,
  error mapping (bad key, 403, 404 model, 429, 503, 504, timeout), and responses that are
  blocked, cut off or empty. Provider-selection logic is tested too.
- **Live LLM** (`test_agent_live.py`): real calls for an explanation, a follow-up, a plan and
  `/agent/text`. Excluded from normal runs; run it with `.venv/bin/python -m pytest -m live -s`.
  The live run against Gemini returned a real explanation and follow-up, then one Gemini 503
  ("high demand") on the plan. Rerun through the running app, all four calls succeeded.