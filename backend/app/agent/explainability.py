"""Model sensitivity analysis for one prediction.

Random Forests have no per-prediction coefficients, so this re-runs the saved pipeline with one
input changed at a time and reports how the bad-risk score moves. It describes how *this model*
responds to its inputs; it is not a causal statement about the applicant.
"""

import pandas as pd

from ml.features import MODEL_FEATURES, NUMERIC_FEATURES

METHOD = (
    "One-at-a-time sensitivity analysis: the saved model is re-run with a single input replaced and "
    "all other inputs held fixed. It shows how the model's score responds to that input. It is not "
    "causal and ignores interactions between changes."
)

# Inputs an applicant could plausibly change for a future application. Only financially sensible
# directions are tried (smaller/shorter loans, fewer credits, higher balances). Categories such as
# "no checking account" or "unknown savings" score as low risk in this 1994 data, but closing an
# account is not advice, so they are never offered as alternatives.
LOWER_IS_SAFER_NUMERIC = {
    "duration_months": [6, 12, 18, 24, 36],
    "installment_rate_pct": [1, 2, 3],
    "existing_credits": [1, 2, 3],
}
CREDIT_AMOUNT_FACTORS = [0.5, 0.75]
IMPROVEMENT_LADDERS = {
    "checking_status": ["lt_0_dm", "0_to_200_dm", "ge_200_dm_or_salary"],
    "savings_status": ["lt_100_dm", "100_to_500_dm", "500_to_1000_dm", "ge_1000_dm"],
}
NO_OTHER_PLANS = ("other_installment_plans", "none")


def _improvements(features):
    for f, options in LOWER_IS_SAFER_NUMERIC.items():
        for v in options:
            if float(v) < float(features[f]):
                yield f, v
    for factor in CREDIT_AMOUNT_FACTORS:
        v = max(250.0, round(float(features["credit_amount"]) * factor, 2))
        if v < float(features["credit_amount"]):
            yield "credit_amount", v
    for f, ladder in IMPROVEMENT_LADDERS.items():
        current = features[f]
        higher = ladder[ladder.index(current) + 1:] if current in ladder else ladder[1:]
        yield from ((f, v) for v in higher)
    f, v = NO_OTHER_PLANS
    if features[f] != v:
        yield f, v


def _score(pipeline, rows):
    frame = pd.DataFrame(rows, columns=MODEL_FEATURES)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return pipeline.predict_proba(frame)[:, 1]


def sensitivity(pipeline, features, typical, threshold, top_n=8):
    """`typical` holds training-set medians / modes (model_info.feature_defaults)."""
    base_row = {f: features[f] for f in MODEL_FEATURES}
    variants = [("baseline", None, None)]

    for f in MODEL_FEATURES:
        typical_value = typical[f]
        if f in NUMERIC_FEATURES:
            if float(features[f]) != float(typical_value):
                variants.append(("typical", f, typical_value))
        elif features[f] != typical_value:
            variants.append(("typical", f, typical_value))

    variants.extend(("what_if", f, v) for f, v in _improvements(features))

    rows = [{**base_row, **({f: v} if f else {})} for _, f, v in variants]
    scores = _score(pipeline, rows)
    baseline = float(scores[0])

    vs_typical, what_if = [], []
    for (kind, f, v), score in zip(variants[1:], scores[1:]):
        entry = {
            "feature": f,
            "submitted_value": features[f],
            "probability_bad_if_changed": round(float(score), 4),
            "change_in_probability_bad": round(float(score) - baseline, 4),
        }
        if kind == "typical":
            vs_typical.append({**entry, "typical_value": v})
        else:
            what_if.append({**entry, "alternative_value": v,
                            "would_cross_threshold": (baseline >= threshold) != (float(score) >= threshold)})

    vs_typical.sort(key=lambda e: -abs(e["change_in_probability_bad"]))
    # best (largest decrease) alternative per actionable feature
    best = {}
    for e in what_if:
        if e["change_in_probability_bad"] < 0 and (
                e["feature"] not in best or e["change_in_probability_bad"] < best[e["feature"]]["change_in_probability_bad"]):
            best[e["feature"]] = e
    return {
        "method": METHOD,
        "baseline_probability_bad": round(baseline, 4),
        "decision_threshold": threshold,
        "inputs_differing_from_typical_applicant": vs_typical[:top_n],
        "largest_score_reductions_for_changeable_inputs": sorted(
            best.values(), key=lambda e: e["change_in_probability_bad"]),
        "note": "Only financially sensible changes were tested (smaller or shorter loans, fewer credits, "
                "higher account balances, no other instalment plans). The comparison with a typical applicant "
                "is descriptive only: some categories (e.g. 'no checking account', 'unknown/no savings') look "
                "low-risk in this 1994 data, which is a historical correlation, not a recommendation.",
    }
