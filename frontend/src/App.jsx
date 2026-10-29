import { useCallback, useEffect, useState } from "react";
import { API_URL, api } from "./api.js";
import AgentPanel from "./components/AgentPanel.jsx";
import History from "./components/History.jsx";
import MetricsOverview from "./components/MetricsOverview.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import PredictionForm from "./components/PredictionForm.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import { humanize, safeHttpUrl } from "./format.js";
import { mine } from "./mine.js";

// A stored prediction (from /predictions/lookup) in the same shape as a /predict response.
function fromHistory(row) {
  return {
    prediction_id: row.id,
    prediction: row.predicted_class ? "bad_credit_risk" : "good_credit_risk",
    predicted_class: row.predicted_class,
    probability_bad: row.probability_bad,
    probability_good: 1 - row.probability_bad,
    decision_threshold: row.decision_threshold,
    model_name: row.model_name,
    model_trained_at: row.model_trained_at,
    warnings: [],
    disclaimer: `Reopened from your prediction history (scored ${new Date(row.created_at).toLocaleString()}).`,
  };
}

export default function App() {
  const [state, setState] = useState({ loading: true, error: null, health: null, metrics: null, info: null, schema: null });
  const [prediction, setPrediction] = useState({ loading: false, result: null, error: null, fieldErrors: {} });
  const [history, setHistory] = useState({ rows: [], error: null });
  const [agentStatus, setAgentStatus] = useState(null);

  const loadHistory = useCallback(() => {
    const items = mine.all().map(({ id, token }) => ({ id, token }));
    if (items.length === 0) {
      setHistory({ rows: [], error: null });
      return;
    }
    api.lookupPredictions(items)
      .then((rows) => setHistory({ rows: rows.slice(0, 10), error: null }))
      .catch((e) => setHistory({ rows: [], error: e.message }));
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        if (!health.model_loaded) {
          setState({ loading: false, error: health.error, health });
          return;
        }
        const [metrics, info, schema] = await Promise.all([api.metrics(), api.modelInfo(), api.schema()]);
        setState({ loading: false, error: null, health, metrics, info, schema });
        loadHistory();
        api.agentStatus().then(setAgentStatus)
          .catch((e) => setAgentStatus({ configured: false, message: `Agent status unavailable: ${e.message}` }));
      } catch (e) {
        setState({ loading: false, error: e.message, health: null });
      }
    })();
  }, [loadHistory]);

  const predict = async (features) => {
    setPrediction({ loading: true, result: null, error: null, fieldErrors: {} });
    try {
      const result = await api.predict(features);
      if (result.prediction_id && result.access_token) mine.add(result.prediction_id, result.access_token);
      setPrediction({ loading: false, result, error: null, fieldErrors: {} });
      loadHistory();
    } catch (e) {
      setPrediction({ loading: false, result: null, error: e.message, fieldErrors: e.fieldErrors || {} });
    }
  };

  const { loading, error, health, metrics, info, schema } = state;

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Credit Risk Scoring</h1>
          <p className="muted">German Credit (UCI Statlog) · scikit-learn model served by FastAPI</p>
        </div>
        <div className={`pill ${health?.model_loaded ? "good" : loading ? "" : "critical"}`}>
          {loading ? "Connecting…" : health?.model_loaded ? "✓ Backend online · model loaded" : "✕ Model unavailable"}
        </div>
      </header>

      {loading && <p className="muted">Loading model metrics from {API_URL}…</p>}
      {error && (
        <div className="banner error" role="alert">
          <strong>Backend problem.</strong> {error}
        </div>
      )}

      {metrics && info && schema && (
        <>
          <MetricsOverview metrics={metrics} modelInfo={info} />
          <div className="two-col">
            <PredictionForm schema={schema} modelInfo={info} onSubmit={predict}
              loading={prediction.loading} serverErrors={prediction.fieldErrors} />
            <div className="stack">
              <PredictionResult {...prediction} />
              <History {...history} onSelect={(row) => setPrediction({ loading: false, result: fromHistory(row), error: null, fieldErrors: {} })} />
            </div>
          </div>
          <AgentPanel prediction={prediction.result} status={agentStatus} />
          <ModelComparison metrics={metrics} />
          <section className="card about">
            <h2>About this model</h2>
            <dl className="facts">
              <dt>Selection criterion</dt><dd>{info.selection_criterion}</dd>
              <dt>Target labels</dt><dd>{info.label_mapping}</dd>
              <dt>Class imbalance</dt><dd>{info.class_imbalance_handling} (dataset bad-risk rate {(info.dataset.bad_rate * 100).toFixed(0)}%)</dd>
              <dt>Probability calibration</dt><dd>{info.probability_calibration}</dd>
              <dt>Excluded features</dt><dd>{info.excluded_features.map(humanize).join(", ")}: kept out of the model as protected attributes (sex, national origin).</dd>
              <dt>Dataset</dt>
              <dd>
                {safeHttpUrl(info.dataset.source)
                  ? <a href={safeHttpUrl(info.dataset.source)} target="_blank" rel="noopener noreferrer">{info.dataset.name}</a>
                  : info.dataset.name} ·
                {" "}{info.dataset.license} · {info.dataset.n_rows} rows · train {info.split.n_train} / test {info.split.n_test}
              </dd>
              <dt>Software</dt><dd>scikit-learn {info.sklearn_version}, Python {info.python_version}</dd>
            </dl>
            <p className="disclaimer">
              The output is the probability of the dataset's "bad credit risk" label, learned from 1,000 historical
              applications from 1994 Germany. It is a default-risk classifier, not a complete real-world credit score.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
