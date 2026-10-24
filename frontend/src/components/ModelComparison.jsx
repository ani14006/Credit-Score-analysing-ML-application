import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { MODEL_LABELS, num, pct } from "../format.js";
import { MODEL_COLOR_KEY, useChartTheme } from "../theme.js";

const METRICS = [
  ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"],
];

function ChartTooltip({ active, payload, label, labelFormatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tt-label">{labelFormatter ? labelFormatter(label) : label}</div>
      {payload.map((p) => (
        <div key={p.dataKey + p.name} className="tt-row">
          <span className="swatch" style={{ background: p.color }} />
          <span>{p.name}</span>
          <strong>{num(p.value)}</strong>
        </div>
      ))}
    </div>
  );
}

// Relative luminance (WCAG) of a #rrggbb colour, to pick legible ink per cell.
function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function ConfusionMatrix({ name, cm, theme }) {
  const max = Math.max(cm.tn, cm.fp, cm.fn, cm.tp);
  const cell = (value, label) => {
    const step = Math.min(theme.seq.length - 1, Math.floor((value / max) * (theme.seq.length - 1)));
    const bg = theme.seq[step];
    return (
      <div className="cm-cell" style={{ background: bg, color: luminance(bg) < 0.25 ? "#fff" : "#0b0b0b" }} title={`${label}: ${value}`}>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    );
  };
  return (
    <figure className="cm">
      <figcaption>{MODEL_LABELS[name]}</figcaption>
      <div className="cm-grid">
        <div />
        <div className="cm-axis">Pred. good</div>
        <div className="cm-axis">Pred. bad</div>
        <div className="cm-axis side">Actual good</div>
        {cell(cm.tn, "True negative")}
        {cell(cm.fp, "False positive")}
        <div className="cm-axis side">Actual bad</div>
        {cell(cm.fn, "False negative")}
        {cell(cm.tp, "True positive")}
      </div>
    </figure>
  );
}

export default function ModelComparison({ metrics }) {
  const theme = useChartTheme();
  const names = Object.keys(metrics.models);
  const barData = METRICS.map(([key, label]) => ({
    metric: label,
    ...Object.fromEntries(names.map((n) => [n, metrics.models[n].test[key]])),
  }));
  const axisProps = { stroke: theme.axis, tick: { fill: theme.muted, fontSize: 12 }, tickLine: false };

  return (
    <section className="card">
      <div className="card-head">
        <h2>Model comparison</h2>
        <p className="muted">
          All values computed by <code>ml/train.py</code> from real predictions on the same held-out test set.
          Each model uses its own cost-optimal threshold, tuned by cross-validation on the training set.
        </p>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th><th>CV cost / applicant</th><th>CV ROC-AUC</th><th>Threshold</th>
              <th>Test accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Test cost</th>
            </tr>
          </thead>
          <tbody>
            {names.map((n) => {
              const { cv, test } = metrics.models[n];
              return (
                <tr key={n} className={n === metrics.selected_model ? "selected" : ""}>
                  <td>
                    <span className="swatch" style={{ background: theme[MODEL_COLOR_KEY[n]] }} />
                    {MODEL_LABELS[n]}{n === metrics.selected_model && <span className="badge">selected</span>}
                  </td>
                  <td>{num(cv.mean_cost_per_applicant)}</td><td>{num(cv.roc_auc)}</td><td>{test.threshold.toFixed(2)}</td>
                  <td>{pct(test.accuracy)}</td><td>{pct(test.precision)}</td><td>{pct(test.recall)}</td>
                  <td>{num(test.f1)}</td><td>{num(test.roc_auc)}</td><td>{test.total_cost}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="charts">
        <figure className="chart">
          <figcaption>Test-set metrics</figcaption>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} barGap={2} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke={theme.grid} />
              <XAxis dataKey="metric" {...axisProps} />
              <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} axisLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: theme.grid, opacity: 0.5 }} />
              <Legend wrapperStyle={{ fontSize: 12, color: theme.textSecondary }} />
              {names.map((n) => (
                <Bar key={n} dataKey={n} name={MODEL_LABELS[n]} fill={theme[MODEL_COLOR_KEY[n]]} radius={[4, 4, 0, 0]} maxBarSize={28} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </figure>

        <figure className="chart">
          <figcaption>ROC curve (test set)</figcaption>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart margin={{ top: 8, right: 12, left: -12, bottom: 12 }}>
              <CartesianGrid stroke={theme.grid} />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps}
                label={{ value: "False positive rate", position: "insideBottom", offset: -6, fill: theme.muted, fontSize: 12 }} />
              <YAxis type="number" dataKey="tpr" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} axisLine={false} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={theme.muted} strokeDasharray="4 4"
                label={{ value: "chance", position: "insideBottomRight", fill: theme.muted, fontSize: 11 }} />
              <Tooltip content={<ChartTooltip labelFormatter={(v) => `FPR ${num(v, 2)} · TPR`} />} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, color: theme.textSecondary }} />
              {names.map((n) => (
                <Line key={n} data={metrics.models[n].test.roc_curve} dataKey="tpr" type="stepAfter" dot={false} strokeWidth={2}
                  name={`${MODEL_LABELS[n]} (AUC ${num(metrics.models[n].test.roc_auc)})`} stroke={theme[MODEL_COLOR_KEY[n]]}
                  activeDot={{ r: 4, stroke: theme.surface, strokeWidth: 2 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </figure>
      </div>

      <h3>Confusion matrices (test set, positive class = bad credit risk)</h3>
      <div className="cms">
        {names.map((n) => <ConfusionMatrix key={n} name={n} cm={metrics.models[n].test.confusion_matrix} theme={theme} />)}
      </div>
    </section>
  );
}
