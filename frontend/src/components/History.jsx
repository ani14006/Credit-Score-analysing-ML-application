import { pct } from "../format.js";

export default function History({ rows, error }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>Prediction history</h2>
        <p className="muted">Persisted by the backend in SQLite (<code>GET /predictions</code>). Most recent 10.</p>
      </div>
      {error && <div className="banner error">{error}</div>}
      {!error && rows.length === 0 && <p className="muted">No predictions stored yet.</p>}
      {rows.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Time</th><th>Amount</th><th>Term</th><th>P(bad)</th><th>Result</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{new Date(r.created_at).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}</td>
                  <td>{r.features.credit_amount.toLocaleString()} DM</td>
                  <td>{r.features.duration_months} mo</td>
                  <td>{pct(r.probability_bad)}</td>
                  <td>
                    <span className={`pill ${r.predicted_class ? "critical" : "good"}`}>
                      {r.predicted_class ? "✕ Bad" : "✓ Good"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
