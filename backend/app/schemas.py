from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.features import MODEL_FEATURES, category_options

_options = category_options()


def _choice(feature):
    # Literal built from the same option list the training data was decoded with.
    return Literal[tuple(_options[feature])]


class CustomerFeatures(BaseModel):
    """Applicant features. Names and types match ml.features.MODEL_FEATURES."""

    model_config = ConfigDict(extra="forbid")

    duration_months: int = Field(ge=1, le=120, description="Loan duration in months")
    credit_amount: float = Field(gt=0, le=100_000, description="Credit amount (Deutsche Mark)")
    installment_rate_pct: int = Field(
        ge=1, le=4, description="Installment rate as % of disposable income, coded 1-4 in the source data")
    residence_since: int = Field(ge=1, le=4, description="Present residence since, coded 1-4")
    age_years: int = Field(ge=18, le=100)
    existing_credits: int = Field(ge=1, le=10, description="Number of existing credits at this bank")
    num_dependents: int = Field(ge=0, le=10, description="People liable to provide maintenance for")

    checking_status: _choice("checking_status")
    credit_history: _choice("credit_history")
    purpose: _choice("purpose")
    savings_status: _choice("savings_status")
    employment_since: _choice("employment_since")
    other_debtors: _choice("other_debtors")
    property: _choice("property")
    other_installment_plans: _choice("other_installment_plans")
    housing: _choice("housing")
    job: _choice("job")
    telephone: _choice("telephone")


assert set(CustomerFeatures.model_fields) == set(MODEL_FEATURES), "API schema drifted from model features"


class PredictionResponse(BaseModel):
    prediction_id: int | None
    prediction: Literal["good_credit_risk", "bad_credit_risk"]
    predicted_class: int
    probability_bad: float
    probability_good: float
    decision_threshold: float
    model_name: str
    model_trained_at: str
    warnings: list[str]
    disclaimer: str
