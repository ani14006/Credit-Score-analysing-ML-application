import { useMemo, useState } from "react";
import { FIELD_LABELS, humanize } from "../format.js";

function validateField(spec, raw) {
  if (spec.enum) return spec.enum.includes(raw) ? null : "Choose a value";
  if (raw === "" || raw === null || raw === undefined) return "Required";
  const value = Number(raw);
  if (!Number.isFinite(value)) return "Must be a number";
  if (spec.type === "integer" && !Number.isInteger(value)) return "Must be a whole number";
  if (spec.minimum !== undefined && value < spec.minimum) return `Must be ≥ ${spec.minimum}`;
  if (spec.exclusiveMinimum !== undefined && value <= spec.exclusiveMinimum) return `Must be > ${spec.exclusiveMinimum}`;
  if (spec.maximum !== undefined && value > spec.maximum) return `Must be ≤ ${spec.maximum}`;
  return null;
}

export default function PredictionForm({ schema, modelInfo, onSubmit, loading, serverErrors }) {
  const numeric = modelInfo.numeric_features;
  const categorical = modelInfo.categorical_features;
  const initial = useMemo(() => {
    const d = modelInfo.feature_defaults || {};
    return Object.fromEntries([...numeric, ...categorical].map((f) => [f, d[f] !== undefined ? String(d[f]) : ""]));
  }, [modelInfo]);

  const [values, setValues] = useState(initial);
  const [errors, setErrors] = useState({});

  const set = (field, value) => {
    setValues((v) => ({ ...v, [field]: value }));
    setErrors((e) => ({ ...e, [field]: null }));
  };

  const submit = (event) => {
    event.preventDefault();
    const nextErrors = {};
    for (const field of [...numeric, ...categorical]) {
      const err = validateField(schema.properties[field], values[field]);
      if (err) nextErrors[field] = err;
    }
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length) return;
    const payload = Object.fromEntries([
      ...numeric.map((f) => [f, Number(values[f])]),
      ...categorical.map((f) => [f, values[f]]),
    ]);
    onSubmit(payload);
  };

  const errorFor = (f) => errors[f] || serverErrors?.[f];

  return (
    <form className="card" onSubmit={submit} noValidate>
      <div className="card-head">
        <h2>Applicant financial information</h2>
        <p className="muted">
          Prefilled with the training-set median / most common value for each feature. Edit before predicting.
        </p>
      </div>
      <div className="form-grid">
        {numeric.map((f) => {
          const spec = schema.properties[f];
          return (
            <label key={f} className={errorFor(f) ? "field invalid" : "field"}>
              <span>{FIELD_LABELS[f] || f}</span>
              <input
                type="number"
                inputMode="decimal"
                step={spec.type === "integer" ? 1 : "any"}
                min={spec.minimum ?? spec.exclusiveMinimum}
                max={spec.maximum}
                value={values[f]}
                onChange={(e) => set(f, e.target.value)}
                aria-invalid={!!errorFor(f)}
              />
              {errorFor(f) && <em className="field-error">{errorFor(f)}</em>}
            </label>
          );
        })}
        {categorical.map((f) => (
          <label key={f} className={errorFor(f) ? "field invalid" : "field"}>
            <span>{FIELD_LABELS[f] || f}</span>
            <select value={values[f]} onChange={(e) => set(f, e.target.value)} aria-invalid={!!errorFor(f)}>
              {schema.properties[f].enum.map((o) => (
                <option key={o} value={o}>{humanize(o)}</option>
              ))}
            </select>
            {errorFor(f) && <em className="field-error">{errorFor(f)}</em>}
          </label>
        ))}
      </div>
      <div className="form-actions">
        <button type="button" className="secondary" onClick={() => { setValues(initial); setErrors({}); }} disabled={loading}>
          Reset
        </button>
        <button type="submit" disabled={loading}>{loading ? "Predicting…" : "Predict credit risk"}</button>
      </div>
    </form>
  );
}
