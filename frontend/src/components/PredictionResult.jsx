import { MODEL_LABELS, pct } from "../format.js";

export default function PredictionResult({ result, error, loading }) {
  if (loading) return <section className="card result"><p className="muted">Running the saved model…</p></section>;
  if (error) {
    return (
      <section className="card result">
        <div className="banner error" role="alert"><strong>Prediction failed.</strong> {error}</div>
      </section>
    );
  }
  if (!result) {
    return (
      <section className="card result">
        <h2>Prediction</h2>
        <p className="muted">Submit the form to score an applicant with the saved model.</p>
      </section>
    );
  }
  const bad = result.predicted_class === 1;
  const p = result.probability_bad;
  return (
    <section className="card result" aria-live="polite">
      <h2>Prediction</h2>
      <div className={`status ${bad ? "critical" : "good"}`}>
        <span className="status-icon" aria-hidden>{bad ? "✕" : "✓"}</span>
        {bad ? "Bad credit risk" : "Good credit risk"}
      </div>
      <div className="prob">
        <div className="prob-value">{pct(p)}</div>
        <div className="muted">model score P(bad credit risk) · uncalibrated, compare to threshold</div>
      </div>
      <div className="meter" role="img" aria-label={`P(bad) ${pct(p)}, threshold ${pct(result.decision_threshold, 0)}`}>
        <div className="meter-fill" style={{ width: `${p * 100}%` }} />
        <div className="meter-threshold" style={{ left: `${result.decision_threshold * 100}%` }}>
          <span>threshold {result.decision_threshold.toFixed(2)}</span>
        </div>
      </div>
      <dl className="facts">
        <dt>P(good credit risk)</dt><dd>{pct(result.probability_good)}</dd>
        <dt>Decision rule</dt><dd>bad if P(bad) ≥ {result.decision_threshold.toFixed(2)}</dd>
        <dt>Model</dt><dd>{MODEL_LABELS[result.model_name] || result.model_name}</dd>
        <dt>Prediction ID</dt><dd>#{result.prediction_id}</dd>
      </dl>
      {result.warnings.length > 0 && (
        <div className="banner warning">
          {result.warnings.map((w) => <div key={w}>⚠ {w}</div>)}
        </div>
      )}
      <p className="disclaimer">{result.disclaimer}</p>
    </section>
  );
}
