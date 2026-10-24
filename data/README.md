# Dataset: Statlog (German Credit Data)

| | |
|---|---|
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Download | https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip |
| Creator | Prof. Dr. Hans Hofmann, Universität Hamburg (1994) |
| DOI | 10.24432/C5NC77 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Files kept | `raw/german.data` (1,000 rows, 20 attributes + label), `raw/german.doc` (original documentation) |
| SHA-256 of `german.data` | `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871` |

Citation: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

The file is stored unmodified. `ml/data.py` decodes the symbolic codes (`A11`, `A43`, …)
into readable values using the mapping in `raw/german.doc`; the mapping lives in `ml/features.py`.
The data are anonymised historical records; no private or personal data was added to this project.

## Target label mapping

| Raw label | Meaning in source | Model target `bad_credit_risk` | API `prediction` |
|---|---|---|---|
| 1 | Good credit risk | 0 | `good_credit_risk` |
| 2 | Bad credit risk  | 1 (positive class) | `bad_credit_risk` |

Class balance: 700 good / 300 bad.

"Bad credit risk" is the dataset's own label for applicants the bank classified as bad risks.
It is **not** a measured default event over a fixed horizon, and a probability of this label is
**not** a complete real-world credit score (no bureau data, no repayment history, 1994 Germany,
1,000 applicants).

## Cost matrix (from `german.doc`)

|                | predicted good | predicted bad |
|---|---|---|
| actual good    | 0 | 1 |
| actual bad     | 5 | 0 |

This cost matrix drives model and threshold selection (see the top-level README).

## Features not used by the model

* `personal_status_sex` — combines marital status and sex.
* `foreign_worker` — national origin.

Both are decoded and kept in the loaded DataFrame for auditing, but are excluded from
`MODEL_FEATURES` and rejected by the API. `age_years` is used; note that age is a protected
attribute in some jurisdictions.
