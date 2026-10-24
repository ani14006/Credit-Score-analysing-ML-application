export const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
};

export const FIELD_LABELS = {
  duration_months: "Loan duration (months)",
  credit_amount: "Credit amount (DM)",
  installment_rate_pct: "Installment rate (1–4)",
  residence_since: "Present residence since (1–4)",
  age_years: "Age (years)",
  existing_credits: "Existing credits at this bank",
  num_dependents: "Number of dependents",
  checking_status: "Checking account status",
  credit_history: "Credit history",
  purpose: "Loan purpose",
  savings_status: "Savings account / bonds",
  employment_since: "Employed since",
  other_debtors: "Other debtors / guarantors",
  property: "Property",
  other_installment_plans: "Other installment plans",
  housing: "Housing",
  job: "Job",
  telephone: "Telephone",
};

const TOKENS = { lt: "<", ge: "≥", to: "–", dm: "DM", tv: "TV", or: "or" };

export function humanize(value) {
  const text = String(value)
    .split("_")
    .map((t) => TOKENS[t] ?? t)
    .join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const num = (v, digits = 3) => Number(v).toFixed(digits);
