# Technical Manual

Deep reference for the payment churn risk pipeline: schema, module
responsibilities, modeling methodology, and worked examples showing
exactly how a policy's raw payment history turns into a risk score and a
recommendation.

See [`../README.md`](../README.md) for the project overview. If you're
learning this codebase from scratch,
[`Code_Walkthrough.md`](Code_Walkthrough.md) teaches it script by script,
in execution order, with exercises — this document is for looking things
up, that one is for reading start to finish.

## Contents

- [Repository layout](#repository-layout)
- [Database schema](#database-schema)
- [Data flow](#data-flow)
- [Module reference](#module-reference)
- [Feature engineering](#feature-engineering)
- [Modeling methodology](#modeling-methodology)
- [Explainability](#explainability)
- [Recommendations](#recommendations)
- [Batch scoring: the Active-only fix](#batch-scoring-the-active-only-fix)
- [Worked examples](#worked-examples)
- [Known limitations](#known-limitations)
- [Development setup & usage](#development-setup--usage)

---

## Repository layout

```
churn/
├── README.md                   # customer/portfolio-facing overview
├── docs/
│   ├── Technical.md            # this file
│   └── Code_Walkthrough.md     # script-by-script course, with exercises
├── diagrams/
│   ├── architecture.html       # interactive component diagram (archify)
│   ├── architecture.png        # static screenshot, embedded in README.md
│   └── dataflow.html           # interactive data-flow diagram (archify)
├── notes/                      # personal scratch notes (gitignored)
├── docker-compose.yml          # local Postgres
├── pyproject.toml
├── db/
│   ├── create_tables.sql       # schema, also the Postgres init script
│   └── db_tables.xlsx          # original schema design spec
├── syntethic_data/
│   ├── generate_data.py        # synthetic data generator
│   ├── tbl_*.csv                # generated raw tables
│   └── policy_profile_debug.csv # hidden ground-truth profile (segmentation validation only)
├── src/churn/
│   ├── churn_db.py              # engine/connection setup
│   ├── seed_db.py               # CSV -> Postgres loader
│   ├── features.py              # raw tables -> per-policy feature table
│   ├── segmentation.py          # unsupervised clustering
│   ├── churn_trainer.py         # model training + persistence
│   ├── evaluation.py            # metrics, calibration, cost-weighted threshold
│   ├── explain.py               # SHAP values, per-policy + aggregate
│   ├── recommend.py             # driver -> action / plain-language mapping
│   ├── churn_predict.py         # batch scoring
│   └── report.py                # Excel report generation
├── models/                      # persisted model + report artifacts (gitignored)
└── tests/                       # one file per module, no live DB required
```

## Database schema

PostgreSQL 16, defined in `db/create_tables.sql`:

```
tbl_customer ──< tbl_policies ──< tbl_invoice ──< tbl_payhistory
                       │
                       └──< tbl_policy_risk_score
```

| Table | Key columns | Notes |
|---|---|---|
| `tbl_customer` | `customer_id` (PK) | name, tax ID |
| `tbl_policies` | `policy_num` (PK), `customer_id` (FK) | `policy_status` ∈ {Active, Cancelled}, `cancel_date` (nullable), `effective_start_date`/`effective_end_date` |
| `tbl_invoice` | `invoice_num` (PK), `policy_num` (FK) | one row per billing installment, `days_arrears_allowed` defines the grace window |
| `tbl_payhistory` | `transaction_id` (PK), `invoice_num` (FK) | one row per payment *attempt* — an invoice can have several (e.g. a failed attempt followed by a successful retry) |
| `tbl_policy_risk_score` | `(policy_num, scored_at)` composite PK | one row per policy per scoring run — see below |

`tbl_policy_risk_score` is the only table added beyond the original
schema:

```sql
CREATE TABLE IF NOT EXISTS tbl_policy_risk_score (
    policy_num          TEXT            NOT NULL,
    scored_at           TIMESTAMP       NOT NULL DEFAULT now(),
    churn_probability   NUMERIC(5, 4),
    segment_label       TEXT,           -- from segmentation.py
    top_factors         JSONB,          -- SHAP top contributors, e.g. [{"feature": "failed_invoice_ratio", "value": 0.41}, ...]
    recommended_action  TEXT,           -- from recommend.py
    CONSTRAINT pk_policy_risk_score PRIMARY KEY (policy_num, scored_at),
    CONSTRAINT fk_risk_score_policy FOREIGN KEY (policy_num)
        REFERENCES tbl_policies (policy_num)
);
```

Keyed on `(policy_num, scored_at)` rather than just `policy_num` so
re-running the batch scorer over time builds a risk trend instead of
overwriting history. `recommended_action` is populated by `recommend.py`
at scoring time; there's no separate table for the driver→action mapping
itself since that's static business logic, not row-level data.

## Data flow

```
generate_data.py --> syntethic_data/*.csv --> seed_db.py --> Postgres (raw tables)
                                                                  │
                                                                  ▼
                                                          features.py (feature table)
                                                          ┌───────┴───────┐
                                                          ▼               ▼
                                                  segmentation.py   churn_trainer.py
                                                  (clusters)        (trains + persists model)
                                                                          │
                                                                          ▼
                                                                    evaluation.py
                                                                    (metrics, threshold)
                                                                          │
                                                                          ▼
                                                                     explain.py
                                                                    (SHAP values)
                                                                          │
                                                                          ▼
                                                              churn_predict.py
                                                     (filters to Active, scores, clusters,
                                                      explains, recommends — one pass)
                                                                          │
                                                            ┌─────────────┴─────────────┐
                                                            ▼                           ▼
                                                 tbl_policy_risk_score            models/feature_importance.csv
                                                            │
                                                            ▼
                                                       report.py
                                                            │
                                                            ▼
                                              models/risk_report_YYYYMMDD.xlsx
```

For an interactive version of this same diagram (pan/zoom, guided
"training path" vs. "scoring path" views), see
[`../diagrams/dataflow.html`](../diagrams/dataflow.html). The system
component view (Postgres, the Python modules, model artifacts, and where
the retention team fits in) is at
[`../diagrams/architecture.html`](../diagrams/architecture.html).

Training (`churn_trainer.py`/`evaluation.py`/`explain.py`'s standalone
runs) always uses the **full** feature table, Cancelled policies
included — the model needs both classes to learn from. Only
`churn_predict.py`'s batch-scoring step filters to Active policies (see
[below](#batch-scoring-the-active-only-fix)).

## Module reference

### `churn_db.py`
Builds a SQLAlchemy engine from `DATABASE_URL` (env var, falls back to
the docker-compose default). Every other module that touches Postgres
goes through `get_engine()`.

### `seed_db.py`
Loads the four synthetic CSVs into Postgres. Idempotent by design —
`ON CONFLICT DO NOTHING` on UUID primary keys, so regenerating more
synthetic data and rerunning this script merges new rows in rather than
duplicating. `upsert_dataframe()` is generic (works for any table via
SQLAlchemy reflection) and is reused by `churn_predict.py` to write
scoring results.

`clean_date_columns()` exists because of a real bug: `df.where(pd.notnull(df),
None)` does **not** turn `NaT` into `None` on a `datetime64` column (the
column can't hold `None`, so pandas silently keeps the internal `NaT`
sentinel, which then leaks into Postgres as a garbage date like
`48113-11-21`). The fix — cast to `object` dtype *before* `.where()` —
has a regression test (`tests/test_seed_db.py`).

### `features.py`
Turns the four raw tables into one row per policy. Key steps:

1. `summarize_payhistory_by_invoice()` — one row per invoice: was it
   missed entirely (no payhistory row at all), did any attempt fail, was
   it eventually paid late, and how many days past
   `days_arrears_allowed` did the successful payment land.
2. `aggregate_policy_features()` — collapses that to one row per policy:
   counts/ratios of missed/failed/late invoices, average/max days late,
   and a "trend delta" (recent-half rate minus early-half rate) for both
   the failed rate and the late rate.
3. `policy_static_features()` — tenure (`effective_end_date` minus
   `effective_start_date`, but `cancel_date` instead of
   `effective_end_date` for Cancelled policies) and the label
   (`is_churned = 1` iff `policy_status == 'Cancelled'`).

`stringify_columns()` is a defensive fix for a real type bug: SQLAlchemy
hands back column names as `quoted_name` (a `str` subclass) for columns
read straight from Postgres, mixed with plain `str` for pandas-computed
columns — `.astype(str)` is a no-op on `quoted_name` (it already passes
pandas' "is this a string" check), so scikit-learn's `ColumnTransformer`
rejected the mixed-type column index outright until this was added.

`BEHAVIOR_FEATURE_COLUMNS` — the canonical list of numeric features — is
exported from here and imported everywhere else (training, segmentation,
explainability) so nothing risks training on a different feature set than
it explains.

### `segmentation.py`
`cluster_policies()` runs `KMeans` (k=3, standardized inputs) on
`BEHAVIOR_FEATURE_COLUMNS` alone — never `is_churned` or `policy_status`.
`label_clusters_by_risk()` then names the clusters "Low/Medium/High risk"
by ranking mean `failed_invoice_ratio` — purely descriptive, after the
fact, not fed back into how the clusters were formed.

`compare_to_hidden_profile()` validates the clustering against
`policy_profile_debug.csv` — the `good`/`at_risk`/`churned` behavior
profile `generate_data.py` assigned when *generating* the synthetic data,
which never enters Postgres or the feature table. Adjusted Rand index on
the full dataset: **0.294**, with the "High risk" cluster mapping almost
entirely onto the hidden `churned` profile (26/26, zero false positives
from `at_risk`/`good`).

### `churn_trainer.py`
`build_pipeline()` bundles a `ColumnTransformer` (standardize the numeric
features, one-hot encode `insurance_product`/`payment_frequency`) with a
classifier — defaults to `class_weight="balanced"` logistic regression
(churn is the minority class, ~24% of policies), or pass
`build_random_forest_classifier()` to swap it for the random forest used
in production. `split_feature_table()` centralizes the stratified 80/20
train/test split so `evaluation.py` and this module always split
identically. `save_pipeline()`/`load_pipeline()` persist via `joblib` to
`models/churn_model.joblib` — the file the batch scorer loads.

### `evaluation.py`
Goes beyond accuracy: precision/recall/F1/ROC-AUC/PR-AUC, a confusion
matrix, a calibration table (predicted vs. observed churn rate by
quantile bucket), and a full threshold sweep. `pick_cost_weighted_threshold()`
picks whichever threshold minimizes
`false_positives × cost_fp + false_negatives × cost_fn` rather than
defaulting to 0.5 — see [Modeling methodology](#modeling-methodology) for
the real numbers. `save_evaluation_report()` persists the result to
`models/evaluation_report.json`/`threshold_sweep.csv`/`calibration.csv`.

### `explain.py`
`fit_random_forest()` trains the random forest on the same split as
`churn_trainer.py`. `compute_shap_values()` runs `shap.TreeExplainer` on
the fitted classifier over the *transformed* (post-`ColumnTransformer`)
feature space, normalizing across a real SHAP API difference — older
versions return a list of per-class arrays from `shap_values()`, current
versions (0.52, what this project runs) return a single
`(n_samples, n_features, n_classes)` array. `explain_policy()` ranks one
policy's contributions by magnitude; `aggregate_feature_importance()`
averages `|SHAP value|` across a batch.

### `recommend.py`
Pure lookup logic — see [Recommendations](#recommendations) for the full
tables.

### `churn_predict.py`
Orchestrates one scoring run: filter to Active policies, score, cluster
for a segment label, explain via SHAP, recommend an action, write to
`tbl_policy_risk_score`, and persist the batch's aggregate feature
importance to `models/feature_importance.csv`. See
[Batch scoring](#batch-scoring-the-active-only-fix) for the filter logic
specifically.

### `report.py`
Reads the latest `tbl_policy_risk_score` batch (joined with
`tbl_policies` for product/status context) and writes
`models/risk_report_YYYYMMDD.xlsx` — no modeling logic, purely
presentation. Two sheets: "Policy Detail" (one row per policy,
color-coded by segment, autofiltered) and "Summary" (totals + two native
Excel bar charts). Does no ML work itself by design — everything it shows
was already computed by `churn_predict.py`.

## Feature engineering

Per policy, as of the reference date:

| Feature | Definition |
|---|---|
| `missed_invoice_ratio` | fraction of invoices with **zero** payhistory rows at all |
| `failed_invoice_ratio` | fraction of invoices with at least one `Failed` payment attempt |
| `late_invoice_ratio` | fraction of invoices ultimately paid, but with status `Paid Late` |
| `avg_days_late` / `max_days_late` | average/max days past `days_arrears_allowed`, computed only over invoices that were actually paid (a missed invoice has no "days late" to measure — excluded, not treated as 0) |
| `trend_failed_rate_delta` / `trend_late_rate_delta` | (failed/late rate in the more recent half of a policy's invoices) − (rate in the earlier half); 0.0 when there are fewer than 2 invoices |
| `tenure_days` | `effective_end_date − effective_start_date`, except for Cancelled policies where it's `cancel_date − effective_start_date` |
| `is_churned` | the label — 1 iff `policy_status == 'Cancelled'` |

Categorical passthroughs: `insurance_product`, `payment_frequency`
(one-hot encoded inside the model pipeline, not here).

## Modeling methodology

Two models were compared on the full 370-policy dataset (stratified
80/20 split, 74 held-out policies):

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic regression (baseline) | 1.000 | 0.944 | 0.997 |
| Random forest (production) | 1.000 | 1.000 | 1.000 |

The random forest was chosen for both the marginally better numbers and
because SHAP's `TreeExplainer` is exact and fast for tree models, versus
the general-purpose `KernelExplainer` a linear model would need.

**Why evaluation isn't just accuracy:** churn is ~24% of the dataset, so
a model that predicts "not churned" for everyone would already score
~76% accuracy while being useless. `evaluation.py` explicitly frames the
two error types by their business cost:

- **False positive** — a policy flagged as at-risk that wouldn't have
  churned. Cost: one wasted retention outreach.
- **False negative** — a policy that churns but the model misses. Cost:
  a lost customer with zero warning — assumed to be worth several times
  more than one wasted outreach (`cost_fn = 5 × cost_fp` by default).

On the held-out split, both the default (0.5) and cost-weighted (0.25)
thresholds land on identical metrics (the model made zero errors either
way on this split) — see `models/evaluation_report.json` for the current
numbers, and `models/threshold_sweep.csv` for the full sweep showing
`expected_cost` at every threshold from 0.05 to 0.95.

## Explainability

SHAP `TreeExplainer` decomposes every prediction into per-feature
contributions. Two views:

- **Per-policy** (`explain_policy()`) — "why is *this* policy at risk?"
  Ranked by `|SHAP value|`, positive = pushed toward risk, negative =
  pushed toward safe.
- **Aggregate** (`aggregate_feature_importance()`) — "what behaviors
  drive risk overall?" Mean `|SHAP value|` across a batch.

**Important interpretive note:** across the currently-Active population,
`tenure_days` dominates the aggregate ranking (mean `|SHAP|` ≈ 0.31,
roughly 3.5× the next feature) — but this is largely a constant-baseline
effect, not genuine differentiation. Every policy's synthetic term is a
fixed 365 days (`effective_end_date = effective_start_date + 365`), so
every Active policy carries essentially the same `tenure_days` value and
gets pulled toward "safe" by roughly the same amount (see the [worked
examples](#worked-examples) below — every one shows a `tenure_days` SHAP
value in the −0.32 to −0.38 range). The real differentiation among
currently-open policies comes from the failed/missed/late behavior
features, which is exactly where you'd want the signal to come from.

## Recommendations

`recommend.py` maps each policy's top *actionable* SHAP driver — skipping
`tenure_days` (outcome-adjacent, see above) and one-hot categoricals, and
requiring a positive SHAP value (only act on something pushing toward
risk) — to a suggested action:

| Driver | Suggested action |
|---|---|
| `missed_invoice_ratio` | Proactive outreach before the next invoice is due — history of invoices going unpaid entirely |
| `failed_invoice_ratio` | Prompt a payment-method update — repeated failures usually mean an expired card or bank decline |
| `late_invoice_ratio` | Offer a flexible due date or grace-period plan |
| `avg_days_late` | Send payment reminders earlier in the billing cycle |
| `max_days_late` | Escalate to a retention call |
| `trend_failed_rate_delta` | Escalate to the retention team — failure rate worsening recently |
| `trend_late_rate_delta` | Proactive check-in call — lateness trending worse recently |
| *(nothing qualifies)* | Monitor — no single actionable driver stands out |

`describe_factors()` uses the same filter but returns up to three
plain-language bullets (e.g. "Repeated failed payment attempts") instead
of a single action sentence — this is what populates the report's "Why
Flagged" column.

**A real limitation:** only the top 3 SHAP factors are stored per policy
(`TOP_FACTORS_PER_POLICY` in `churn_predict.py`). If two drivers are
similarly significant, the smaller one can occasionally be edged out of
the top 3 by a lower-magnitude categorical feature — see
[`POL-8C41550C`](#pol-8c41550c) below, where `failed_invoice_ratio` (tied
at 0.33 with `missed_invoice_ratio`) didn't make the stored list.

## Batch scoring: the Active-only fix

**The bug:** `churn_predict.py` originally scored every policy in the
feature table regardless of `policy_status`. A Cancelled policy's outcome
has already happened, so a "churn risk" score for one isn't a
prediction — it's the model recognizing its own training label
(`is_churned = 1`) reflected back through that policy's own behavior
features. In the pre-fix batch, **164 of 175** policies flagged "at risk"
(94%) were already Cancelled or Expired — not actionable, and actively
misleading in a report meant to drive retention outreach.

**The fix:** `filter_active_policies()` restricts the feature table to
`policy_status == 'Active'` immediately after `build_feature_table()`,
before anything gets scored, clustered, or written. Training
(`churn_trainer.py`, `evaluation.py`, `explain.py`'s standalone runs)
still uses the full dataset including Cancelled policies — the model
needs both classes to learn from; only the batch-scoring *output* is
restricted.

```python
def filter_active_policies(feature_table: pd.DataFrame) -> pd.DataFrame:
    return feature_table[feature_table["policy_status"] == "Active"].reset_index(drop=True)
```

**A related data note:** the synthetic generator's fixed historical date
window meant most non-churned policies had already run past their
365-day term and were labeled `Expired`, functionally identical to
`Active` (`is_churned = 0` either way, no code branches on the
distinction). To give the corrected pipeline a realistic book to work
with, all 245 `Expired` policies were relabeled `Active` directly in
Postgres (status column only — no dates touched, so no behavior feature
is affected). Live counts went from 245 Expired / 88 Cancelled / 37
Active to 0 / 88 / **282** Active.

## Worked examples

Four policies, picked from a real scored batch, tracing the full path
from raw payment rows to the final report row. All are Active policies
(`tenure_days = 365`, the full nominal term).

### POL-1C1CDD7E — 20.5%, Medium risk

| Installment | Arrears window | Outcome |
|---|---|---|
| 1 | 60 days | **missed** (no payment record at all) |
| 2 | 90 days | Paid Late — 29 days past window |
| 3 | 30 days | Paid on time |
| 4 | 60 days | Paid Late — 15 days past window |
| 5 | 90 days | **missed** |

No `Failed` transactions anywhere — every attempted payment succeeded,
just late or absent. Features: `missed_invoice_ratio = 0.40`,
`late_invoice_ratio = 0.40`, `failed_invoice_ratio = 0.0`,
`avg_days_late = 14.67`.

SHAP: `tenure_days` −0.32, `missed_invoice_ratio` **+0.13**,
`failed_invoice_ratio` −0.05 (correctly safe — it's genuinely 0). →
*"History of missed payments"* → **proactive outreach before the next
invoice**. Coherent: this customer skips bills, doesn't fail payments —
outreach before the next due date is the right lever, not a card update.

### POL-63B2B68F — 22.5%, High risk

2 invoices: #1 clean, #2 **Failed** then retried and paid 8 days later.
`failed_invoice_ratio = 0.5`, everything else 0.

SHAP: `tenure_days` −0.38, `failed_invoice_ratio` **+0.14**,
`missed_invoice_ratio` −0.07. → *"Repeated failed payment attempts"* →
**prompt a payment-method update**. A single resolved card decline is the
textbook case for this action.

### POL-8C41550C

3 invoices: #1 missed, #2 Paid Late (16 days over a 30-day window), #3
**Failed** with no retry on record. `missed_invoice_ratio = 0.33`,
`failed_invoice_ratio = 0.33`, `late_invoice_ratio = 0.33` — three
different problems, evenly split.

SHAP top-3: `tenure_days` −0.36, `missed_invoice_ratio` **+0.08**,
`payment_frequency_Semi-Annual` −0.03. → *"History of missed payments"* →
**proactive outreach**. Note: `failed_invoice_ratio` is numerically tied
with `missed_invoice_ratio` here but got edged out of the stored top-3 by
a smaller categorical effect — see the [limitation](#recommendations)
noted above. In a full explanation (not just the stored top-3), the
failed-payment signal would also be worth surfacing.

### POL-1958DB2B

13 invoices — a long history. **6 separate `Failed` attempts**
(installments 1, 5, 8, 11, 12, 13), all eventually retried and paid; 2
missed outright; zero late. `failed_invoice_ratio = 0.46`,
`missed_invoice_ratio = 0.15`, `trend_failed_rate_delta = 0.38`
(failures worsening recently).

SHAP: `tenure_days` −0.38, `failed_invoice_ratio` **+0.09**,
`missed_invoice_ratio` −0.05 (interesting — pushes *safe* despite being
nonzero; a normal effect of how tree-ensemble SHAP values reflect
feature interactions, not a bug). → *"Repeated failed payment attempts"*
→ **prompt a payment-method update**. With 6 failures across a year,
this is the clearest-cut case of the four.

## Known limitations

- **`tenure_days` is a near-constant among Active policies** — see
  [Explainability](#explainability). Its dominance in the aggregate
  ranking reflects the synthetic data's fixed 1-year term, not a
  meaningful behavioral driver among open policies.
- **Only the top 3 SHAP factors are persisted per policy** — an equally
  significant secondary driver can occasionally be left off; see
  `POL-8C41550C` above.
- **`pandas` is pinned below 3.0** (`pyproject.toml`) — installing
  `streamlit` once pulled in pandas 3.0.6 with no upper bound, which
  changed `Index` construction behavior enough to break a real regression
  test. The pin stays as a guard even after `streamlit` was later removed.
- **Segmentation reclusters per batch** — `churn_predict.py` calls
  `cluster_policies()` fresh on whatever feature table it's given, so
  "High/Medium/Low risk" segment boundaries are relative to *that batch's*
  population, not a fixed global scale. After the Active-only fix, this
  means segments are relative to the current active book only.
- **Synthetic data, not production data** — every number in this
  document comes from a generated dataset (`syntethic_data/generate_data.py`),
  not real customer records.

## Development setup & usage

```bash
docker compose up -d              # Postgres on localhost:5432
pip install -e ".[dev]"           # project + pytest/faker
cp .env.example .env              # adjust DATABASE_URL if needed
python syntethic_data/generate_data.py --customers 120 --seed 42
python -m churn.seed_db
pytest                            # 60 tests, no live DB required
```

Rerunning `generate_data.py` with a new seed/customer count and rerunning
`seed_db.py` merges more data in — the seed script is intentionally
idempotent (`ON CONFLICT DO NOTHING` on UUID keys) rather than a one-time
bootstrap, since growing the synthetic dataset over time is the expected
workflow for this project.

To train, evaluate, score, and report from scratch:

```bash
python -m churn.evaluation      # trains, evaluates, persists the model
python -m churn.churn_predict   # scores Active policies, writes to Postgres
python -m churn.report          # generates the dated Excel report
```

That last step produces `models/risk_report_YYYYMMDD.xlsx` — open it
directly, no server involved.
