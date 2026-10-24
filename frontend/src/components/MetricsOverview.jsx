import { MODEL_LABELS, num, pct } from "../format.js";

export default function MetricsOverview({ metrics, modelInfo }) {
  const selected = metrics.models[metrics.selected_model];
  const t = selected.test;
  const tiles = [
    ["ROC-AUC", num(t.roc_auc), "Ranking quality, threshold-free"],
    ["Recall (bad risk)", pct(t.recall), "Share of truly bad applicants caught"],
    ["Precision (bad risk)", pct(t.precision), "Flagged applicants who were truly bad"],
    ["F1-score", num(t.f1), "Harmonic mean of precision & recall"],
    ["Accuracy", pct(t.accuracy), "All correct predictions"],
    ["Misclassification cost", `${t.total_cost}`, `5×FN + 1×FP over ${t.n_samples} applicants`],
  ];
  return (
    <section className="card">
      <div className="card-head">
        <h2>Selected model: {MODEL_LABELS[metrics.selected_model]}</h2>
        <p className="muted">
          Held-out test set ({modelInfo.split.n_test} applicants, never used for training or selection) ·
          decision threshold P(bad) ≥ {t.threshold.toFixed(2)} · trained {new Date(metrics.trained_at).toLocaleString()}
        </p>
      </div>
      <div className="tiles">
        {tiles.map(([label, value, hint]) => (
          <div className="tile" key={label}>
            <div className="tile-label">{label}</div>
            <div className="tile-value">{value}</div>
            <div className="tile-hint">{hint}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
