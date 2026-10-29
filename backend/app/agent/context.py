"""Builds the verified context the LLM receives. Everything here comes from the backend:
the stored prediction, the saved model and its evaluation artifacts. Anything the user adds
is kept in a separate, explicitly unverified section."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.explainability import sensitivity
from ml.features import CATEGORY_CODES, MODEL_FEATURES

# Meaning of each model input, from data/raw/german.doc (amounts are 1994 Deutsche Mark).
FEATURE_INFO = {
    "duration_months": "Loan duration in months",
    "credit_amount": "Credit amount requested, in Deutsche Mark (DM)",
    "installment_rate_pct": "Instalment as a share of disposable income, coded 1 (lowest) to 4 (highest)",
    "residence_since": "Years at present residence, coded 1 to 4",
    "age_years": "Age in years",
    "existing_credits": "Number of existing credits at this bank",
    "num_dependents": "Number of people the applicant financially supports",
    "checking_status": "Balance of the checking account at the bank",
    "credit_history": "Repayment history category recorded by the bank",
    "purpose": "Purpose of the loan",
    "savings_status": "Savings account / bonds balance",
    "employment_since": "Time in present employment",
    "other_debtors": "Co-applicant or guarantor on the loan",
    "property": "Most valuable property owned",
    "other_installment_plans": "Instalment plans with other lenders",
    "housing": "Housing situation",
    "job": "Job category",
    "telephone": "Telephone registered in the applicant's name",
}

VALUE_TEXT = {
    "checking_status": {"lt_0_dm": "overdrawn (< 0 DM)", "0_to_200_dm": "0 to 200 DM",
                        "ge_200_dm_or_salary": "200 DM or more, or salary paid in for 1+ year",
                        "no_checking_account": "no checking account"},
    "credit_history": {"no_credits_or_all_paid": "no credits taken / all credits paid back duly",
                       "all_paid_this_bank": "all credits at this bank paid back duly",
                       "existing_paid_duly": "existing credits paid back duly till now",
                       "past_delay": "delay in paying off in the past",
                       "critical_or_other_credits": "critical account / other credits existing elsewhere"},
    "savings_status": {"lt_100_dm": "< 100 DM", "100_to_500_dm": "100 to 500 DM",
                       "500_to_1000_dm": "500 to 1000 DM", "ge_1000_dm": "1000 DM or more",
                       "unknown_or_none": "unknown / no savings account"},
}

LIMITATIONS = [
    "The model predicts the 'good' or 'bad' credit-risk label that a German bank assigned to 1,000 "
    "applicants in 1994 (UCI Statlog German Credit dataset). It is not a credit bureau score such as "
    "CIBIL, FICO or SCHUFA and cannot be converted into one.",
    "probability_bad is an uncalibrated model score: class-weighted training makes it overstate risk "
    "(on the test set the mean predicted value was about 0.41 while the observed bad rate was 0.30). "
    "Compare it with the decision threshold; do not read it as a literal default probability.",
    "Amounts are in 1994 Deutsche Mark. Do not convert them to current currencies or treat them as "
    "today's income or debt levels.",
    "The model was trained and evaluated on 800 / 200 applicants; its test ROC-AUC is about 0.79, so "
    "it misclassifies a substantial share of applicants.",
    "Sex/marital status and foreign-worker status were excluded from the model. Age is used.",
    "Some categories look low-risk in this 1994 data for reasons unrelated to good financial habits "
    "(for example 'no checking account' and 'unknown / no savings account'). That is not a reason to "
    "close accounts, reduce savings or take on more credit.",
    "The sensitivity analysis shows how the model responds to inputs; it does not show what caused "
    "this applicant's outcome or what a lender today would decide.",
]

NOT_COLLECTED = [
    "income or salary", "monthly expenses", "current outstanding balances", "credit card limits and "
    "balances (so credit utilisation cannot be computed)", "payment history beyond the bank's coded "
    "category", "any credit bureau score (CIBIL, FICO, etc.)", "interest rates or EMI amounts on "
    "existing loans", "missed or late payments in recent months",
]


class UserFinancialContext(BaseModel):
    """Optional details the user chooses to share. Never verified by the backend."""

    model_config = ConfigDict(extra="forbid")

    goal: str | None = Field(None, max_length=300)
    currency: str | None = Field(None, max_length=8)
    monthly_net_income: float | None = Field(None, ge=0, le=1e9)
    monthly_essential_expenses: float | None = Field(None, ge=0, le=1e9)
    monthly_emi_total: float | None = Field(None, ge=0, le=1e9)
    total_outstanding_debt: float | None = Field(None, ge=0, le=1e10)
    total_credit_card_limit: float | None = Field(None, ge=0, le=1e10)
    total_credit_card_balance: float | None = Field(None, ge=0, le=1e10)
    emergency_savings: float | None = Field(None, ge=0, le=1e10)
    missed_payments_last_12_months: int | None = Field(None, ge=0, le=365)
    reported_credit_score: int | None = Field(None, ge=0, le=1000)
    reported_credit_score_type: Literal["CIBIL", "FICO", "Experian", "Equifax", "TransUnion", "SCHUFA", "other"] | None = None
    notes: str | None = Field(None, max_length=1000)

    def provided(self):
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}


def describe_features(features):
    out = []
    for f in MODEL_FEATURES:
        value = features[f]
        entry = {"name": f, "meaning": FEATURE_INFO[f], "value": value}
        if f in CATEGORY_CODES:
            entry["value_meaning"] = VALUE_TEXT.get(f, {}).get(value, value.replace("_", " "))
        out.append(entry)
    return out


def build_context(prediction, store, user_context=None):
    """`prediction` is a row loaded from the database; `store` is the loaded ModelStore."""
    bundle, info, metrics = store.bundle, store.model_info, store.metrics
    selected = metrics["models"][metrics["selected_model"]]["test"]
    analysis = sensitivity(bundle["pipeline"], prediction["features"], info["feature_defaults"],
                           prediction["decision_threshold"])
    label = "bad_credit_risk" if prediction["predicted_class"] == 1 else "good_credit_risk"
    return {
        "prediction": {
            "prediction_id": prediction["id"],
            "created_at": prediction["created_at"],
            "model": prediction["model_name"],
            "model_trained_at": prediction["model_trained_at"],
            "predicted_label": label,
            "probability_bad": round(prediction["probability_bad"], 4),
            "decision_threshold": prediction["decision_threshold"],
            "decision_rule": "bad_credit_risk if probability_bad >= decision_threshold",
            "saved_model_still_current": prediction["model_trained_at"] == bundle["trained_at"],
        },
        "submitted_features": describe_features(prediction["features"]),
        "model_sensitivity_analysis": analysis,
        "model_evaluation_on_held_out_test_set": {
            "n_applicants": selected["n_samples"],
            "accuracy": round(selected["accuracy"], 3),
            "precision_bad_risk": round(selected["precision"], 3),
            "recall_bad_risk": round(selected["recall"], 3),
            "f1": round(selected["f1"], 3),
            "roc_auc": round(selected["roc_auc"], 3),
            "brier_score": round(selected["calibration"]["brier_score"], 4),
        },
        "limitations": LIMITATIONS,
        "information_not_collected": NOT_COLLECTED,
        "user_supplied_unverified": user_context.provided() if user_context else {},
    }
