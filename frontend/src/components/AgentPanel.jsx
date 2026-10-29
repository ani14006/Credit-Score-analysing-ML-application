import { useEffect, useState } from "react";
import { api } from "../api.js";
import { FIELD_LABELS, humanize, pct } from "../format.js";
import { mine } from "../mine.js";

const BASIS = {
  observed: { label: "Observed", hint: "Directly in your submitted data or the model output" },
  model_analysis: { label: "Model analysis", hint: "From the model's sensitivity analysis or test metrics" },
  user_reported: { label: "You reported", hint: "From details you added; not verified" },
  general_education: { label: "General guidance", hint: "Standard financial education, not specific to your data" },
  assumption: { label: "Assumption", hint: "An inference the agent made; check it" },
};

const SUGGESTED = [
  "Why did the model classify my application this way?",
  "What should I focus on first?",
  "How do I manage my existing debt?",
  "Explain credit utilization in simple terms.",
  "What information would you need to give more specific advice?",
];

const CONTEXT_FIELDS = [
  ["goal", "Your main financial goal", "text"],
  ["currency", "Currency (e.g. INR)", "text"],
  ["monthly_net_income", "Monthly take-home income", "number"],
  ["monthly_essential_expenses", "Monthly essential expenses", "number"],
  ["monthly_emi_total", "Total monthly EMIs", "number"],
  ["total_outstanding_debt", "Total outstanding debt", "number"],
  ["total_credit_card_limit", "Total credit card limit", "number"],
  ["total_credit_card_balance", "Current credit card balance", "number"],
  ["emergency_savings", "Emergency savings", "number"],
  ["missed_payments_last_12_months", "Missed payments (last 12 months)", "integer"],
  ["reported_credit_score", "Your credit score, if known", "integer"],
];
const SCORE_TYPES = ["CIBIL", "FICO", "Experian", "Equifax", "TransUnion", "SCHUFA", "other"];

function Basis({ value }) {
  const b = BASIS[value] || { label: value, hint: "" };
  return <span className={`basis basis-${value}`} title={b.hint}>{b.label}</span>;
}

function Points({ title, items, empty }) {
  return (
    <div className="agent-block">
      <h4>{title}</h4>
      {items.length === 0 ? <p className="muted">{empty}</p> : (
        <ul className="points">
          {items.map((p, i) => (
            <li key={i}>
              <div className="point-head"><strong>{p.title}</strong><Basis value={p.basis} /></div>
              <p>{p.detail}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Plain-text answer renderer: paragraphs and "- " bullets only, rendered as React text (no HTML).
function AnswerText({ text }) {
  const blocks = text.split(/\n{2,}/);
  return blocks.map((block, i) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-•]\s+/.test(l))) {
      return <ul key={i}>{lines.map((l, j) => <li key={j}>{l.replace(/^\s*[-•]\s+/, "")}</li>)}</ul>;
    }
    return <p key={i}>{block}</p>;
  });
}

function ErrorBox({ message, onRetry }) {
  return (
    <div className="banner error" role="alert">
      {message}
      {onRetry && <button type="button" className="link-button" onClick={onRetry}>Retry</button>}
    </div>
  );
}

function ContextForm({ value, onChange }) {
  const set = (k, v) => onChange({ ...value, [k]: v });
  return (
    <details className="context-form">
      <summary>Add more financial details (optional)</summary>
      <p className="muted">
        The model only knows the 18 fields above. Anything you add here is sent to the agent as
        user-reported and unverified. Leave fields empty if you prefer not to share them.
      </p>
      <div className="form-grid">
        {CONTEXT_FIELDS.map(([key, label, type]) => (
          <label key={key} className="field">
            <span>{label}</span>
            <input type={type === "text" ? "text" : "number"} min={type === "text" ? undefined : 0}
              step={type === "integer" ? 1 : "any"} maxLength={key === "goal" ? 300 : key === "currency" ? 8 : undefined}
              value={value[key] ?? ""} onChange={(e) => set(key, e.target.value)} />
          </label>
        ))}
        <label className="field">
          <span>Score type</span>
          <select value={value.reported_credit_score_type ?? ""} onChange={(e) => set("reported_credit_score_type", e.target.value)}>
            <option value="">Not specified</option>
            {SCORE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="field wide">
          <span>Anything else (max 1000 characters)</span>
          <textarea rows={2} maxLength={1000} value={value.notes ?? ""} onChange={(e) => set("notes", e.target.value)} />
        </label>
      </div>
    </details>
  );
}

function toUserContext(raw) {
  const out = {};
  for (const [key, , type] of CONTEXT_FIELDS) {
    const v = raw[key];
    if (v === undefined || v === "") continue;
    out[key] = type === "text" ? v.trim() : Number(v);
  }
  if (raw.reported_credit_score_type) out.reported_credit_score_type = raw.reported_credit_score_type;
  if (raw.notes?.trim()) out.notes = raw.notes.trim();
  return Object.keys(out).length ? out : null;
}

function Evidence({ evidence }) {
  const a = evidence.model_sensitivity_analysis;
  const reported = Object.entries(evidence.user_supplied_unverified || {});
  return (
    <details className="evidence">
      <summary>Evidence and limitations (computed by the backend, not by the AI)</summary>
      <div className="evidence-grid">
        <div>
          <h4>Model output</h4>
          <dl className="facts">
            <dt>Label</dt><dd>{humanize(evidence.prediction.predicted_label)}</dd>
            <dt>Score P(bad)</dt><dd>{pct(evidence.prediction.probability_bad)} (uncalibrated)</dd>
            <dt>Threshold</dt><dd>{evidence.prediction.decision_threshold}</dd>
            <dt>Test ROC-AUC</dt><dd>{evidence.model_evaluation_on_held_out_test_set.roc_auc}</dd>
          </dl>
          {reported.length > 0 && (
            <>
              <h4>You reported (unverified)</h4>
              <dl className="facts">
                {reported.map(([k, v]) => [<dt key={`${k}-k`}>{humanize(k)}</dt>, <dd key={`${k}-v`}>{String(v)}</dd>])}
              </dl>
            </>
          )}
        </div>
        <div className="wide">
          <h4>How the model's score responds to changes</h4>
          <p className="muted small">{a.method}</p>
          {a.largest_score_reductions_for_changeable_inputs.length === 0 ? (
            <p className="muted small">None of the tested changes lowered the score.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Input</th><th>Yours</th><th>Tested</th><th>Score change</th></tr></thead>
                <tbody>
                  {a.largest_score_reductions_for_changeable_inputs.map((e) => (
                    <tr key={e.feature}>
                      <td>{FIELD_LABELS[e.feature] || e.feature}</td>
                      <td>{typeof e.submitted_value === "string" ? humanize(e.submitted_value) : e.submitted_value}</td>
                      <td>{typeof e.alternative_value === "string" ? humanize(e.alternative_value) : e.alternative_value}</td>
                      <td>{(e.change_in_probability_bad * 100).toFixed(1)} pts</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="muted small">{a.note}</p>
        </div>
        <div className="wide">
          <h4>Limitations</h4>
          <ul className="small">{evidence.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
          <h4>Not collected by this application</h4>
          <p className="small">{evidence.information_not_collected.join(" · ")}</p>
        </div>
      </div>
    </details>
  );
}

function Explanation({ e }) {
  return (
    <div className="explanation">
      <p className="agent-headline">{e.headline}</p>
      <p>{e.prediction_meaning}</p>
      <div className="agent-columns">
        <Points title="Financial factors to review" items={e.factors_to_review} empty="None identified." />
        <Points title="Positive habits in your data" items={e.positive_habits} empty="The supplied data doesn't show clear positives." />
        <Points title="Risks and missing information" items={e.risks_and_missing_information} empty="None identified." />
      </div>
      <div className="agent-block">
        <h4>Prioritised steps</h4>
        <ol className="steps">
          {e.prioritized_steps.map((s, i) => (
            <li key={i}>
              <div className="point-head">
                <span className={`prio prio-${s.priority}`}>{s.priority}</span>
                <strong>{s.title}</strong><Basis value={s.basis} />
              </div>
              <p>{s.detail}</p>
              <p className="muted small">Why it matters: {s.why_it_matters}</p>
            </li>
          ))}
        </ol>
      </div>
      <div className="agent-block">
        <h4>Action plan</h4>
        <p>{e.action_plan_summary}</p>
      </div>
      <div className="agent-block">
        <h4>Uncertainty and limitations</h4>
        <ul>{e.uncertainty_and_limitations.map((u, i) => <li key={i}>{u}</li>)}</ul>
      </div>
    </div>
  );
}

function Planner({ session, token, onPlan }) {
  const [horizon, setHorizon] = useState(30);
  const [goals, setGoals] = useState("");
  const [state, setState] = useState({ loading: false, error: null });
  const [taskError, setTaskError] = useState(null);
  const plan = session.plan;

  const generate = async () => {
    setState({ loading: true, error: null });
    try {
      onPlan(await api.createPlan(session.session_id, token, horizon, goals.trim()));
      setState({ loading: false, error: null });
    } catch (e) {
      setState({ loading: false, error: e.message });
    }
  };

  const withTask = (p, id, done) => {
    const tasks = p.tasks.map((t) => (t.id === id ? { ...t, done } : t));
    const completed = tasks.filter((t) => t.done).length;
    return { ...p, tasks, progress: { completed, total: tasks.length, percent: Math.round((100 * completed) / tasks.length) } };
  };

  // Optimistic: tick immediately, then replace with the saved plan (or revert on failure).
  const toggle = async (task) => {
    setTaskError(null);
    const before = plan;
    onPlan(withTask(plan, task.id, !task.done));
    try {
      onPlan(await api.updateTask(session.session_id, token, task.id, !task.done));
    } catch (e) {
      onPlan(before);
      setTaskError(`Couldn't save that change: ${e.message}`);
    }
  };

  const weeks = plan ? [...new Set(plan.tasks.map((t) => t.week))].sort((a, b) => a - b) : [];
  return (
    <section className="agent-section" aria-labelledby="plan-title">
      <h3 id="plan-title">Personalised improvement plan</h3>
      <div className="plan-controls">
        <label className="field">
          <span>Plan length</span>
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} disabled={state.loading}>
            <option value={30}>30 days</option><option value={60}>60 days</option><option value={90}>90 days</option>
          </select>
        </label>
        <label className="field grow">
          <span>Goals for this plan (optional)</span>
          <input type="text" maxLength={300} value={goals} onChange={(e) => setGoals(e.target.value)}
            placeholder="e.g. stop relying on my overdraft" disabled={state.loading} />
        </label>
        <button type="button" onClick={generate} disabled={state.loading}>
          {state.loading ? "Generating plan…" : plan ? "Generate a new plan" : "Generate plan"}
        </button>
      </div>
      {state.error && <ErrorBox message={state.error} onRetry={generate} />}
      {!plan && !state.loading && !state.error && (
        <p className="muted">No plan yet. The agent will build one from your prediction, any details you added and your questions.</p>
      )}
      {plan && (
        <div className="plan">
          <div className="plan-head">
            <div>
              <strong>{plan.title}</strong>
              <p className="muted small">{plan.horizon_days}-day plan · starts {plan.start_date}{plan.goals ? ` · goal: ${plan.goals}` : ""}</p>
            </div>
            <div className="progress" aria-label={`${plan.progress.completed} of ${plan.progress.total} tasks done`}>
              <div className="progress-bar"><div style={{ width: `${plan.progress.percent}%` }} /></div>
              <span>{plan.progress.completed}/{plan.progress.total} done ({plan.progress.percent}%)</span>
            </div>
          </div>
          <p>{plan.overview}</p>
          {taskError && <ErrorBox message={taskError} />}
          {weeks.map((w) => (
            <div key={w} className="plan-week">
              <h4>Week {w}</h4>
              <ul className="tasks">
                {plan.tasks.filter((t) => t.week === w).map((t) => (
                  <li key={t.id} className={t.done ? "done" : ""}>
                    <label className="task">
                      <input type="checkbox" checked={t.done} onChange={() => toggle(t)} />
                      <span>
                        <span className="point-head">
                          <strong>{t.title}</strong>
                          <span className={`prio prio-${t.priority}`}>{t.priority}</span>
                          <Basis value={t.basis} />
                          <span className="muted small">by {t.target_date}</span>
                        </span>
                        <span className="task-text">{t.description}</span>
                        <span className="muted small task-text">Why it matters: {t.why_it_matters}</span>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <p className="disclaimer">Educational plan generated by an AI model. It cannot guarantee any credit-score change, loan approval or financial outcome.</p>
        </div>
      )}
    </section>
  );
}

function Chat({ session, token, onMessages }) {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState({ loading: false, error: null, pending: null });

  const send = async (q) => {
    const text = (q ?? question).trim();
    if (!text) return;
    setState({ loading: true, error: null, pending: text });
    try {
      const r = await api.askAgent(session.session_id, token, text);
      onMessages(r.messages);
      setQuestion("");
      setState({ loading: false, error: null, pending: null });
    } catch (e) {
      setState({ loading: false, error: e.message, pending: text });
    }
  };

  return (
    <section className="agent-section" aria-labelledby="chat-title">
      <h3 id="chat-title">Ask the agent</h3>
      <div className="chips">
        {SUGGESTED.map((s) => (
          <button key={s} type="button" className="chip" onClick={() => send(s)} disabled={state.loading}>{s}</button>
        ))}
        <button type="button" className="chip" disabled={state.loading}
          onClick={() => send("Can you create a 30-day improvement plan?")}>Can you create a 30-day improvement plan?</button>
      </div>
      <div className="chat" aria-live="polite">
        {session.messages.length === 0 && !state.pending && <p className="muted">No questions yet.</p>}
        {session.messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="bubble-role">{m.role === "user" ? "You" : "Agent"}</div>
            {m.role === "user" ? <p>{m.text}</p> : <AnswerText text={m.text} />}
          </div>
        ))}
        {state.loading && (
          <>
            <div className="bubble user"><div className="bubble-role">You</div><p>{state.pending}</p></div>
            <p className="muted">Agent is thinking…</p>
          </>
        )}
      </div>
      {state.error && <ErrorBox message={state.error} onRetry={() => send(state.pending)} />}
      <form className="ask" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <textarea rows={2} maxLength={1000} placeholder="Ask a follow-up question about this result…" value={question}
          onChange={(e) => setQuestion(e.target.value)} disabled={state.loading}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
        <button type="submit" disabled={state.loading || !question.trim()}>{state.loading ? "Asking…" : "Ask"}</button>
      </form>
    </section>
  );
}

export default function AgentPanel({ prediction, status }) {
  const predictionId = prediction?.prediction_id;
  const token = predictionId ? mine.token(predictionId) : null;
  const [rawContext, setRawContext] = useState({});
  const [session, setSession] = useState(null);
  const [state, setState] = useState({ loading: false, error: null });

  useEffect(() => {
    setSession(null);
    setState({ loading: false, error: null });
    const existing = predictionId ? mine.sessionId(predictionId) : null;
    if (existing && token) {
      setState({ loading: true, error: null });
      api.getAgentSession(existing, token)
        .then((s) => { setSession(s); setState({ loading: false, error: null }); })
        .catch(() => setState({ loading: false, error: null })); // expired by retention: start fresh
    }
  }, [predictionId, token]);

  const analyse = async () => {
    setState({ loading: true, error: null });
    try {
      const s = await api.startAgent(predictionId, token, toUserContext(rawContext));
      mine.setSession(predictionId, s.session_id);
      setSession(s);
      setState({ loading: false, error: null });
    } catch (e) {
      setState({ loading: false, error: e.message });
    }
  };

  return (
    <section className="card agent" aria-labelledby="agent-title">
      <div className="card-head agent-title-row">
        <div>
          <h2 id="agent-title">AI Financial Wellness Agent</h2>
          <p className="muted">
            An AI model explains your result using data verified by the backend. It cannot change the
            prediction, and its guidance is general education, not professional financial advice.
          </p>
        </div>
        {status && (
          <span className={`pill ${status.configured ? "good" : "critical"}`}>
            {status.configured ? `✓ ${status.model}` : "✕ AI not configured"}
          </span>
        )}
      </div>

      {status && !status.configured && <div className="banner error" role="alert">{status.message}</div>}
      {!prediction && <p className="muted">Run a prediction above, then ask the agent to explain it.</p>}
      {prediction && !token && (
        <p className="muted">This prediction was not stored, so the agent can't analyse it. Prediction history may be disabled on the backend.</p>
      )}

      {prediction && token && !session && (
        <div className="agent-start">
          <ContextForm value={rawContext} onChange={setRawContext} />
          <button type="button" onClick={analyse} disabled={state.loading || !status?.configured}>
            {state.loading ? "Agent is reading the verified prediction…" : "Analyse this result with AI"}
          </button>
          {state.error && <ErrorBox message={state.error} onRetry={analyse} />}
        </div>
      )}

      {session && (
        <>
          <section className="agent-section" aria-labelledby="explain-title">
            <h3 id="explain-title">Explanation <span className="muted small">· prediction #{session.prediction_id} · {session.model}</span></h3>
            <Explanation e={session.explanation} />
            <Evidence evidence={session.evidence} />
          </section>
          <Planner session={session} token={token} onPlan={(plan) => setSession((s) => ({ ...s, plan }))} />
          <Chat session={session} token={token} onMessages={(messages) => setSession((s) => ({ ...s, messages }))} />
        </>
      )}
    </section>
  );
}
