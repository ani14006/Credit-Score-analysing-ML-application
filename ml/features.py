"""Single source of truth for the feature schema.

Both the training script and the API import from here, so the column names,
dtypes and allowed category values are identical at training and inference.
Codes (A11, A12, ...) come from data/raw/german.doc and are decoded into
readable values.
"""

RAW_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target_raw",
]

CATEGORY_CODES = {
    "checking_status": {
        "A11": "lt_0_dm",
        "A12": "0_to_200_dm",
        "A13": "ge_200_dm_or_salary",
        "A14": "no_checking_account",
    },
    "credit_history": {
        "A30": "no_credits_or_all_paid",
        "A31": "all_paid_this_bank",
        "A32": "existing_paid_duly",
        "A33": "past_delay",
        "A34": "critical_or_other_credits",
    },
    "purpose": {
        "A40": "car_new",
        "A41": "car_used",
        "A42": "furniture_equipment",
        "A43": "radio_tv",
        "A44": "domestic_appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "lt_100_dm",
        "A62": "100_to_500_dm",
        "A63": "500_to_1000_dm",
        "A64": "ge_1000_dm",
        "A65": "unknown_or_none",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "lt_1_year",
        "A73": "1_to_4_years",
        "A74": "4_to_7_years",
        "A75": "ge_7_years",
    },
    "personal_status_sex": {
        "A91": "male_divorced_separated",
        "A92": "female_div_sep_married",
        "A93": "male_single",
        "A94": "male_married_widowed",
        "A95": "female_single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co_applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real_estate",
        "A122": "savings_agreement_or_life_insurance",
        "A123": "car_or_other",
        "A124": "unknown_or_none",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for_free",
    },
    "job": {
        "A171": "unemployed_unskilled_nonresident",
        "A172": "unskilled_resident",
        "A173": "skilled_employee",
        "A174": "management_self_employed_highly_qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes_registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
]

# personal_status_sex encodes sex and foreign_worker encodes national origin.
# Both are kept in the loaded dataset for auditing but are deliberately NOT
# used as model inputs.
EXCLUDED_FEATURES = ["personal_status_sex", "foreign_worker"]

CATEGORICAL_FEATURES = [
    c for c in CATEGORY_CODES if c not in EXCLUDED_FEATURES
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "bad_credit_risk"
# Raw label 1 = good, 2 = bad (german.doc). We model the bad class as positive.
TARGET_MAPPING = {1: 0, 2: 1}
CLASS_LABELS = {0: "good_credit_risk", 1: "bad_credit_risk"}

# Cost matrix shipped with the dataset (german.doc section 8):
# predicting "good" for a truly bad customer (false negative) costs 5,
# predicting "bad" for a truly good customer (false positive) costs 1.
COST_FALSE_NEGATIVE = 5
COST_FALSE_POSITIVE = 1


def category_options():
    """Allowed values for each categorical model feature."""
    return {c: sorted(CATEGORY_CODES[c].values()) for c in CATEGORICAL_FEATURES}
