# Code Walkthrough: A Guided Course Through the Churn Pipeline

This document teaches the codebase one script at a time, in the order the
pipeline actually runs. It assumes basic Python (functions, dicts, lists)
and a little pandas, but explains every SQL, statistics, and machine
learning concept as it comes up — you don't need a data science background
to follow along.

**How to use this:** read it start to finish, in order. Each lesson
builds on the last. Code blocks are copied verbatim from the real files in
`src/churn/`, so you can always cross-reference the actual source.

**Our running example:** to make this concrete instead of abstract, every
lesson traces the same real policy — **`POL-1C1CDD7E`**, a Home insurance
policy on Quarterly payments — through that stage of the pipeline. By
[Lesson 11](#lesson-11--following-one-policy-through-everything) you'll
have watched this one row of data transform all the way from raw payment
history to a line in an Excel report.

Here's its raw story, which we'll refer back to constantly:

| Installment | Arrears window | What happened |
|---|---|---|
| 1 | 60 days | **Missed** — no payment record at all |
| 2 | 90 days | Paid, but **29 days late** |
| 3 | 30 days | Paid on time |
| 4 | 60 days | Paid, but **15 days late** |
| 5 | 90 days | **Missed** — no payment record at all |

Notice: this policy never had a single *failed* payment — it just skips
some bills and pays others late. Keep that in mind; it'll matter later.

## Contents

- [Lesson 0: The Big Picture](#lesson-0-the-big-picture)
- [Lesson 1: Talking to the Database — `churn_db.py`](#lesson-1-talking-to-the-database--churn_dbpy)
- [Lesson 2: Loading Data In — `seed_db.py`](#lesson-2-loading-data-in--seed_dbpy)
- [Lesson 3: Turning History into Features — `features.py`](#lesson-3-turning-history-into-features--featurespy)
- [Lesson 4: Finding Natural Groups — `segmentation.py`](#lesson-4-finding-natural-groups--segmentationpy)
- [Lesson 5: Teaching the Model — `churn_trainer.py`](#lesson-5-teaching-the-model--churn_trainerpy)
- [Lesson 6: Grading the Model Honestly — `evaluation.py`](#lesson-6-grading-the-model-honestly--evaluationpy)
- [Lesson 7: Explaining Predictions — `explain.py`](#lesson-7-explaining-predictions--explainpy)
- [Lesson 8: Turning Insight into Action — `recommend.py`](#lesson-8-turning-insight-into-action--recommendpy)
- [Lesson 9: Putting It Together — `churn_predict.py`](#lesson-9-putting-it-together--churn_predictpy)
- [Lesson 10: Delivering the Result — `report.py`](#lesson-10-delivering-the-result--reportpy)
- [Lesson 11: Following One Policy Through Everything](#lesson-11--following-one-policy-through-everything)
- [Glossary](#glossary)

---

## Lesson 0: The Big Picture

Before touching code, here's the shape of the whole system:

```
generate_data.py --> CSVs --> seed_db.py --> Postgres (raw tables)
                                                    │
                                                    ▼
                                            features.py (one row per policy)
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                              segmentation.py           churn_trainer.py
                              (unsupervised clusters)   (supervised classifier)
                                                                │
                                                    ┌───────────┴───────────┐
                                                    ▼                       ▼
                                            evaluation.py            explain.py
                                            (how good is it?)        (why does it say that?)
                                                                            │
                                                                            ▼
                                                                  churn_predict.py
                                                          (score every open policy)
                                                                            │
                                                                ┌───────────┴───────────┐
                                                                ▼                       ▼
                                                        recommend.py            (write to Postgres)
                                                        (what to do?)                   │
                                                                                        ▼
                                                                                  report.py
                                                                                        │
                                                                                        ▼
                                                                          risk_report_YYYYMMDD.xlsx
```

Ten scripts, each with one job. **Two big ideas hold the whole thing
together, and you'll see both repeatedly:**

1. **Training and scoring are different questions.** Training a model
   needs examples of policies that *did* churn — so it always uses every
   policy, cancelled ones included. Scoring is asking "is this policy
   about to churn?" — which only makes sense for a policy that's still
   open. You'll see this split explicitly in Lesson 9.
2. **Every module does one thing.** `features.py` never trains a model.
   `churn_trainer.py` never reads raw payment tables. This isn't just
   tidiness — it's why `evaluation.py` and `explain.py` can both reuse
   `churn_trainer.py`'s exact train/test split without copying code, and
   why you can test `features.py`'s math without a database at all.

---

## Lesson 1: Talking to the Database — `churn_db.py`

**What you'll learn:** what a database "engine" is, and why connection
details live in environment variables instead of in the code.

This is the smallest file in the project, and a good place to start
because everything else depends on it.

```python
DEFAULT_DATABASE_URL = "postgresql+psycopg://churn:churn@localhost:5432/churn"


def get_database_url() -> str:
    """Return the DB connection string, from env or the local docker default."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    """Build a SQLAlchemy engine for the churn database."""
    return create_engine(get_database_url())
```

**Concept: connection strings.** `postgresql+psycopg://churn:churn@localhost:5432/churn`
packs five things into one string: the database type (`postgresql`), the
driver to use (`psycopg`, a Python library that speaks Postgres's wire
protocol), a username and password (`churn:churn`), a host and port
(`localhost:5432`), and a database name (`churn`). Every one of those
pieces matches `docker-compose.yml`, which is what actually starts the
Postgres server with those exact credentials.

**Concept: an "engine."** In SQLAlchemy (the library this project uses to
talk to Postgres from Python), an `Engine` isn't a single open connection
— it's a factory that hands out and manages a pool of connections as
needed. You create one engine per process and reuse it everywhere, rather
than opening a fresh connection for every query.

**Why `os.getenv` instead of hardcoding the URL?** Hardcoding
`localhost:5432` would work on your laptop and break the moment this ran
anywhere else (a CI pipeline, a colleague's machine, a real server). Every
other module in this project calls `get_engine()` rather than building
its own connection — one function, one place to change if the database
ever moves.

`load_dotenv()` at the top of the file reads a `.env` file in the project
root (if one exists) and loads its contents as environment variables
before `get_database_url()` ever runs. That's how `DATABASE_URL` gets
set without you having to `export` it in your shell every time.

### Exercises

1. What would `get_database_url()` return if you set
   `DATABASE_URL=postgresql+psycopg://someone@example.com/other_db` in
   your `.env` file? Why doesn't `churn_db.py` need to change to support
   that?
2. Every other module calls `get_engine()` rather than constructing its
   own `create_engine(...)` call. What would go wrong (or just get
   annoying) if `features.py` and `seed_db.py` each hardcoded their own
   copy of the connection string?

---

## Lesson 2: Loading Data In — `seed_db.py`

**What you'll learn:** idempotent loading (safe to rerun), what an
"upsert" is, and a real bug about dates that made it into this exact
codebase.

### The date-cleaning function (full function — it's short, and it hides a real trap)

```python
def clean_date_columns(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """
    Swap NaT/NaN for real None, and convert parsed dates to plain python
    date objects. Pulled out on its own (rather than inline in load_csv)
    so it's easy to unit test without touching a csv or a live db.
    """
    # pandas leaves NaT for blanks (e.g. cancel_date on non-cancelled policies).
    # gotcha: df.where(pd.notnull(df), None) on its own does NOT turn NaT into
    # None - a datetime64 column can't hold None, so it silently keeps NaT,
    # which then leaks its raw internal sentinel value to postgres as a
    # garbage date. Casting to object dtype FIRST avoids that.
    df = df.astype(object).where(pd.notnull(df), None)

    # swap remaining Timestamps for plain python date objects (cleaner than
    # handing psycopg a pandas Timestamp)
    for col in date_cols:
        df[col] = df[col].map(lambda v: v.date() if v is not None else None)

    return df
```

**Concept: `NaT`.** pandas has a special "missing value" marker for
dates called `NaT` (Not a Time), the date equivalent of `NaN`. Most
policies in this project haven't been cancelled, so their `cancel_date`
column is blank in the CSV — pandas reads that blank as `NaT`.

**The bug, and why it's worth knowing about.** The obvious way to turn
missing values into `None` (so Postgres sees a real `NULL` instead of
garbage) is `df.where(pd.notnull(df), None)`. That works fine for text
and number columns. It does **not** work for a `datetime64` column,
because that column's data type physically cannot hold a Python `None` —
so pandas silently keeps its internal `NaT` sentinel instead of replacing
it. That sentinel is really just a very large integer under the hood, and
when it got handed to Postgres unconverted, it showed up in the database
as a real but nonsensical date: `48113-11-21`. This actually happened
during this project's development. The fix is the first line of the
function above: cast the whole column to generic `object` dtype **before**
calling `.where(...)` — once the column can hold arbitrary Python objects
instead of only datetime64 values, `None` sticks the way you'd expect.

This is a genuinely useful lesson beyond this one project: **whenever you
convert missing values in a typed column, check what the column's dtype
allows the missing marker to actually be.** The bug lives on as a
regression test in `tests/test_seed_db.py`, specifically checking that a
blank date never comes back as `NaT` disguised as a real date.

### The upsert function

```python
def upsert_dataframe(engine: Engine, table_name: str, df: pd.DataFrame) -> int:
    """Insert every row in df, skipping ones that already exist (same primary key)."""
    if df.empty:
        return 0

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    rows = df.to_dict(orient="records")
    stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    stmt = stmt.returning(table.primary_key.columns.values()[0])

    with engine.begin() as conn:
        result = conn.execute(stmt)
        inserted_ids = result.fetchall()

    return len(inserted_ids)
```

**Concept: idempotent.** An idempotent operation gives you the same end
result no matter how many times you run it. `seed_all()` (the function
that calls this one for each of the four raw tables) is deliberately
*not* a one-time bootstrap script — the plan for this project is to
occasionally generate more synthetic data and rerun the seed step to grow
the dataset. That only works safely if rerunning it with data you've
already loaded doesn't create duplicates.

**Concept: upsert.** "Upsert" = update-or-insert. This particular upsert
is actually the simpler "insert-or-skip" variant: `on_conflict_do_nothing()`
tells Postgres "if a row with this primary key already exists, just skip
it instead of erroring." Since every ID in this project's synthetic data
is a UUID (generated fresh every run of `generate_data.py`), collisions
essentially never happen on a fresh batch — the conflict handling here is
a safety net for the case where you accidentally load the exact same CSV
twice.

**Why check `returning(...)` instead of just trusting the row count?**
When you `INSERT ... ON CONFLICT DO NOTHING` many rows at once, the
"rows affected" count Postgres normally reports isn't reliable for
telling you how many were *actually new* versus silently skipped. Asking
Postgres to `RETURNING` the primary key of whichever rows really got
inserted, then counting those, is the trustworthy way to report "12 new
rows" back to whoever's running the script.

**`autoload_with=engine`** is worth calling out too: instead of hardcoding
column names for each table, `Table(table_name, metadata, autoload_with=engine)`
asks SQLAlchemy to look at Postgres's actual schema and figure out the
columns itself. That's why `upsert_dataframe()` is generic — it works for
any of the four raw tables, and later (Lesson 9) it gets reused for a
completely different table, `tbl_policy_risk_score`, with zero changes.

### Exercises

1. Run `clean_date_columns()` yourself on a tiny DataFrame with one row
   where a date column is `None`. What type is the value in the result —
   is it still a pandas `Timestamp`, or something else?
2. `TABLES_IN_LOAD_ORDER` lists tables in a specific order: customer,
   policies, invoice, payhistory. What would happen if you loaded
   `tbl_invoice` before `tbl_policies`? (Hint: look at the foreign keys in
   `db/create_tables.sql`.)
3. Why does `upsert_dataframe()` return early with `0` when `df.empty` is
   true, instead of just letting the insert statement run on an empty
   list of rows?

---

## Lesson 3: Turning History into Features — `features.py`

**What you'll learn:** feature engineering — turning raw transaction-level
records into one summary row per policy that a model can actually learn
from.

This is the most important file to understand well, because everything
downstream (segmentation, training, explaining) works on its output, not
on the raw tables.

### Step 1 — one row per invoice

```python
def summarize_payhistory_by_invoice(invoices: pd.DataFrame, payhistory: pd.DataFrame) -> pd.DataFrame:
    """
    One row per invoice: was it missed entirely, did it ever fail, did it end
    up paid late, and how many days past the arrears window did it take.
    """
    merged = payhistory.merge(
        invoices[["invoice_num", "invoice_start_date", "days_arrears_allowed"]],
        on="invoice_num",
    )

    # days beyond the arrears window a payment landed on - only meaningful for
    # actual successful payments, a Failed row has no "it arrived on X" date to judge
    merged["days_late"] = (
        (merged["pay_transaction_date"] - merged["invoice_start_date"]).dt.days
        - merged["days_arrears_allowed"]
    ).clip(lower=0)
    merged.loc[merged["pay_transaction_status"] == "Failed", "days_late"] = pd.NA

    per_invoice = merged.groupby("invoice_num").agg(
        failed_attempts=("pay_transaction_status", lambda s: (s == "Failed").sum()),
        was_paid_late=("pay_transaction_status", lambda s: (s == "Paid Late").any()),
        max_days_late=("days_late", "max"),
    )

    summary = invoices[["invoice_num", "policy_num", "installment_num"]].merge(
        per_invoice, on="invoice_num", how="left"
    )
    summary["failed_attempts"] = summary["failed_attempts"].fillna(0).astype(int)
    summary["was_paid_late"] = summary["was_paid_late"].fillna(False)
    summary["was_missed"] = ~summary["invoice_num"].isin(payhistory["invoice_num"])
    summary["had_failed_attempt"] = summary["failed_attempts"] > 0
    return summary
```

**Concept: `merge`.** This is pandas' name for a SQL `JOIN`. Line 1 joins
every payment attempt in `payhistory` to the invoice it belongs to, so
each payment row also carries its invoice's start date and grace-period
length (`days_arrears_allowed`).

**Why `.clip(lower=0)`?** `days_late` is computed as "when it was paid"
minus "when the grace period ended." If a payment lands *before* the
grace period expires, that subtraction goes negative — which isn't
meaningfully "days late," it's "not late at all." `.clip(lower=0)` floors
any negative value at zero.

**Why null out `days_late` for `Failed` rows?** A failed payment attempt
never actually landed — there's no "arrival date" to measure lateness
from. Setting `days_late` to `pd.NA` (pandas' generic missing-value
marker) for those rows means the later `.agg(max_days_late=...)` step
correctly ignores them rather than treating a failure as "0 days late."

**Concept: `groupby().agg()`.** This is one of pandas' core patterns: group
rows by some key (here, `invoice_num`), then compute one or more summary
statistics per group. The `agg(...)` call computes three things per
invoice in one pass: how many failed attempts it had, whether it was ever
marked "Paid Late," and the worst (`max`) lateness among its payment
attempts.

**`was_missed`** isn't computed from `payhistory` at all — it's `True` for
any invoice whose `invoice_num` doesn't appear in `payhistory` at all.
That's the key distinction this project draws between "missed" (nobody
ever tried to pay) and "failed" (someone tried and it didn't go through).

**Applying this to `POL-1C1CDD7E`:** here's exactly what this function
produces for our running example's five invoices:

| installment | failed_attempts | was_paid_late | max_days_late | was_missed |
|---|---|---|---|---|
| 1 | 0 | False | *(NA)* | **True** |
| 2 | 0 | **True** | 29 | False |
| 3 | 0 | False | 0 | False |
| 4 | 0 | **True** | 15 | False |
| 5 | 0 | False | *(NA)* | **True** |

Notice `failed_attempts` is `0` across the board — this policy's story is
entirely about missed and late payments, never a failed charge.

### Step 2 — the trend-delta helper

```python
def _trend_delta(chronological_flags: pd.Series) -> float:
    """
    Recent-half rate minus early-half rate for a 0/1 series already sorted by
    installment order. 0.0 when there aren't at least 2 invoices to compare
    (not enough history for a trend to mean anything).
    """
    n = len(chronological_flags)
    if n < 2:
        return 0.0
    split = math.ceil(n / 2)
    early_rate = chronological_flags.iloc[:split].mean()
    recent_rate = chronological_flags.iloc[split:].mean()
    return float(recent_rate - early_rate)
```

This answers "is this behavior getting worse?" — split a policy's
invoices into an early half and a recent half (in installment order), and
subtract the early rate from the recent rate. A positive number means
things are getting worse; negative means improving; zero means no change
(or not enough history to tell — `n < 2` guards against that).

**Worked for `POL-1C1CDD7E`'s lateness trend:** 5 invoices, so
`split = ceil(5/2) = 3`. Early half = installments 1, 2, 3 → `was_paid_late`
= `[False, True, False]` → rate = 1/3 ≈ 0.333. Recent half = installments
4, 5 → `[True, False]` → rate = 0.5. Trend delta = 0.5 − 0.333 = **+0.167**
— a mild worsening. (Its *failure*-rate trend is 0, since it never failed
at all, early or late.)

### Step 3 — collapsing to one row per policy

```python
def aggregate_policy_features(invoice_summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-invoice summary down to one row per policy."""

    def _per_policy(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("installment_num")
        n = len(group)
        missed = int(group["was_missed"].sum())
        failed = int(group["had_failed_attempt"].sum())
        late = int(group["was_paid_late"].sum())
        return pd.Series(
            {
                "total_invoices": n,
                "missed_invoice_count": missed,
                "missed_invoice_ratio": missed / n,
                "failed_invoice_count": failed,
                "failed_invoice_ratio": failed / n,
                "late_invoice_count": late,
                "late_invoice_ratio": late / n,
                "avg_days_late": group["max_days_late"].mean(skipna=True),
                "max_days_late": group["max_days_late"].max(skipna=True),
                "trend_failed_rate_delta": _trend_delta(group["had_failed_attempt"].astype(float)),
                "trend_late_rate_delta": _trend_delta(group["was_paid_late"].astype(float)),
            }
        )

    features = invoice_summary.groupby("policy_num").apply(_per_policy, include_groups=False)
    features[["avg_days_late", "max_days_late"]] = features[["avg_days_late", "max_days_late"]].fillna(0.0)
    return features.reset_index()
```

**Ratios, not raw counts.** `missed_invoice_ratio = missed / n` rather
than just `missed`. A policy with 2 missed invoices out of 5 is behaving
very differently from one with 2 missed out of 50 — dividing by the total
makes policies with different numbers of invoices comparable on the same
0–1 scale, which matters a lot once we get to modeling.

**Why `skipna=True`, and why fill the result with `0.0` afterward?** A
policy where every single invoice was either missed or failed has no
"successfully-timed payment" at all — `max_days_late` would be `NA` for
every one of its invoices, so `.mean(skipna=True)` on an all-`NA` group
returns `NA` too. Filling that with `0.0` (Line "no lateness observed")
is a deliberate choice: it says "we have no evidence this policy pays
late," which is more honest than dropping the policy from the dataset
entirely.

**For `POL-1C1CDD7E`,** this produces:
`total_invoices=5`, `missed_invoice_ratio=0.4` (2/5),
`failed_invoice_ratio=0.0` (0/5), `late_invoice_ratio=0.4` (2/5),
`avg_days_late = mean(29, 0, 15) = 14.67` (the two missed invoices'
`NA` values are skipped, not averaged in as zero),
`max_days_late=29`, `trend_failed_rate_delta=0.0`,
`trend_late_rate_delta=0.167`.

### Step 4 — the label, and a genuinely subtle date choice

```python
def policy_static_features(policies: pd.DataFrame) -> pd.DataFrame:
    """Policy-level fields that don't need any payhistory rollup: tenure + the label."""
    df = policies.copy()
    # cancelled policies "end" at cancel_date, everyone else at their normal effective_end_date
    end_date = df["cancel_date"].where(df["policy_status"] == "Cancelled", df["effective_end_date"])
    df["tenure_days"] = (pd.to_datetime(end_date) - pd.to_datetime(df["effective_start_date"])).dt.days
    df["is_churned"] = (df["policy_status"] == "Cancelled").astype(int)
    return df[[
        "policy_num", "customer_id", "insurance_product",
        "payment_frequency", "policy_status", "tenure_days", "is_churned",
    ]]
```

**`is_churned` is the label** — the thing the model in Lesson 5 will
learn to predict. It's `1` for a Cancelled policy, `0` for everything
else. This is the only place in the whole pipeline where that ground
truth gets defined.

**`tenure_days` — read this one carefully, because it comes back to bite
us later.** For a Cancelled policy, "how long did it last?" should be
measured to the day it actually cancelled (`cancel_date`), not to its
original scheduled end date — a policy cancelled after 3 months lasted 3
months, not a full year. For every other policy, there's no cancellation,
so we use the normal `effective_end_date`. `pandas.Series.where(cond, other)`
is the tool for this: it keeps the original value where `cond` is true and
substitutes `other` where it's false — here, keep `cancel_date` where the
policy is `Cancelled`, otherwise fall back to `effective_end_date`.

For `POL-1C1CDD7E` (Active, never cancelled): tenure_days =
`effective_end_date (2024-03-31) − effective_start_date (2023-04-01)` =
**365 days** — the full nominal one-year term.

**Hold that thought.** In Lesson 7 you'll see this exact feature turn out
to be a near-constant for every Active policy (they all get a fixed
365-day term), which turns out to matter a lot for how to interpret the
model's explanations.

### Step 5 — a type bug worth knowing about

```python
def stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    pd.read_sql_table hands back column names as sqlalchemy's `quoted_name`
    (a str subclass) for columns sourced straight from postgres, while
    pandas-computed columns end up plain str - a mix scikit-learn now
    rejects outright when a DataFrame with both is fed into a pipeline.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df
```

This looks unnecessary — aren't all column names just strings? Not quite.
When pandas reads a table straight from Postgres via SQLAlchemy, the
column names come back as instances of `quoted_name`, a SQLAlchemy class
that happens to be a subclass of `str` (so it behaves like a string in
almost every way) but isn't *actually* the plain `str` type. Columns that
this project computes itself in pandas (like `missed_invoice_ratio`) are
genuine `str`. Mixing both types in one DataFrame's column index is
invisible to pandas — but scikit-learn's `ColumnTransformer` (Lesson 5)
checks the column index's type strictly and refuses to run on a mixed
one. The fix looks almost too simple to be right — `.astype(str)` alone
doesn't actually work here, because `quoted_name` already satisfies
pandas' "is this a string" check and gets left alone. Rebuilding the
column list with a real `str(c)` call for every column forces genuine
`str` instances. This is a real bug this project's tests guard against
(`tests/test_features.py`).

### Tying it together

```python
def build_feature_table(engine: Engine | None = None) -> pd.DataFrame:
    """Build the full one-row-per-policy feature table straight from postgres."""
    engine = engine or get_engine()
    tables = load_raw_tables(engine)

    invoice_summary = summarize_payhistory_by_invoice(tables["tbl_invoice"], tables["tbl_payhistory"])
    payment_features = aggregate_policy_features(invoice_summary)
    static = policy_static_features(tables["tbl_policies"])

    combined = static.merge(payment_features, on="policy_num", how="left")
    count_and_ratio_cols = [c for c in payment_features.columns if c != "policy_num"]
    combined[count_and_ratio_cols] = combined[count_and_ratio_cols].fillna(0.0)
    return stringify_columns(combined)
```

This is the function every other module calls. Notice the shape of the
whole file: four small, independently-testable functions (each doing one
transformation), combined by one function at the bottom that wires them
together in order. That pattern repeats throughout this codebase.

### Exercises

1. By hand, compute `late_invoice_ratio` and `avg_days_late` for a
   hypothetical policy with 4 invoices where invoice 1 was missed,
   invoices 2 and 3 were paid on time, and invoice 4 was paid 10 days
   late. Then check your answer by writing a tiny pandas script using
   `summarize_payhistory_by_invoice()` and `aggregate_policy_features()`.
2. `_trend_delta()` returns `0.0` when there are fewer than 2 invoices.
   Why is that safer than, say, returning `None` or raising an error?
3. `BEHAVIOR_FEATURE_COLUMNS` is defined once at the top of this file and
   imported by `segmentation.py`, `churn_trainer.py`, and `explain.py`.
   What bug could happen if each of those files instead hardcoded its own
   copy of that list?

---

## Lesson 4: Finding Natural Groups — `segmentation.py`

**What you'll learn:** unsupervised learning — finding structure in data
*without* telling the algorithm the right answer — and how to sanity-check
that it actually found something real.

```python
def cluster_policies(
    feature_table: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add a `cluster` column (raw KMeans label, 0..n_clusters-1) to the feature
    table. Standardizes first since these columns are on wildly different
    scales (ratios 0-1 vs. tenure_days in the hundreds).
    """
    matrix = feature_table[SEGMENTATION_FEATURES]
    scaled = StandardScaler().fit_transform(matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled)

    result = feature_table.copy()
    result["cluster"] = labels
    return result
```

**Concept: unsupervised learning.** Segmentation never looks at
`is_churned` at all — only at `SEGMENTATION_FEATURES` (the same behavior
columns from Lesson 3). It's finding groups of *similar-behaving*
policies purely from their features, with no "right answer" to check
against while it's running. That's the "unsupervised" part: no labels
involved in forming the groups.

**Concept: standardization, and why it matters here.** `missed_invoice_ratio`
lives on a 0–1 scale. `tenure_days` lives in the hundreds. KMeans measures
"similarity" as literal geometric distance between points — without
standardizing first, `tenure_days` would completely dominate that
distance calculation just because its numbers are bigger, drowning out
the ratio columns almost entirely. `StandardScaler().fit_transform(matrix)`
rescales every column to have a mean of 0 and a standard deviation of 1,
putting them all on equal footing before KMeans ever sees them.

**Concept: KMeans.** Given a number of groups to find (`n_clusters=3`
here), KMeans repeatedly assigns each point to its nearest of 3 "center"
points, then recomputes each center as the average of the points assigned
to it, until the assignments stop changing. `n_init=10` runs that whole
process 10 times from different random starting centers and keeps the
best result — KMeans can get stuck in a mediocre local solution depending
on where it starts, so trying several starting points guards against
that. `random_state=42` makes the "random" part reproducible: same input,
same output, every time you run it.

```python
def label_clusters_by_risk(df_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw cluster ids (0/1/2, meaningless on their own) to human-readable
    risk labels, ranked by mean failed_invoice_ratio.
    """
    severity_rank = (
        df_with_cluster.groupby("cluster")["failed_invoice_ratio"].mean().sort_values().index
    )
    risk_names = ["Low risk", "Medium risk", "High risk"]
    names = risk_names + [f"Risk tier {i}" for i in range(3, len(severity_rank))]
    cluster_to_label = dict(zip(severity_rank, names))

    result = df_with_cluster.copy()
    result["segment_label"] = result["cluster"].map(cluster_to_label)
    return result
```

KMeans hands back cluster IDs `0`, `1`, `2` — arbitrary numbers with no
inherent meaning (cluster `0` isn't "low risk" just because it's numbered
first). This function names them *after the fact*, purely for
readability: rank the three clusters by their average `failed_invoice_ratio`
and call the lowest one "Low risk," the highest "High risk." Crucially,
this naming happens **after** clustering is done — it never feeds back
into how the groups were formed, so it can't bias the clustering itself.

**Validating against ground truth the model never saw.** This project's
synthetic data generator (`syntethic_data/generate_data.py`) secretly
assigns each customer a hidden behavior profile — `good`, `at_risk`, or
`churned` — used only to *generate* realistic-looking data, and writes it
to a debug file that's never loaded into Postgres or shown to any model.

```python
def compare_to_hidden_profile(
    segmented: pd.DataFrame, hidden_profile: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    merged = segmented.merge(hidden_profile, on="policy_num", how="inner")
    crosstab = pd.crosstab(merged["segment_label"], merged["hidden_profile"])
    ari = adjusted_rand_score(merged["hidden_profile"], merged["cluster"])
    return crosstab, ari
```

**Concept: adjusted Rand index (ARI).** A single number summarizing how
well two groupings of the same items agree, corrected so that a purely
random grouping scores close to 0 and a perfect match scores 1. This
project's clustering scored **0.294** against the hidden profile — real
structure, not noise, though far from a perfect 1-to-1 recovery (which
would actually be a red flag, since unsupervised clustering on 3-way
generated data isn't expected to perfectly reconstruct the exact
generative labels). The "High risk" cluster specifically lined up almost
perfectly with the hidden `churned` profile.

**Why is this check even here?** It's easy to convince yourself an
unsupervised method "found something meaningful" when really it just
found noise. Having a hidden answer key — that the clustering algorithm
itself never gets to see — turns "I think this makes sense" into an
actual, falsifiable check.

### Exercises

1. Try running `cluster_policies()` with `n_clusters=5` instead of 3 on
   the real feature table. What does `label_clusters_by_risk()` name the
   4th and 5th clusters? Why?
2. Remove `StandardScaler()` from `cluster_policies()` (just pass the raw
   `matrix` straight to KMeans) and compare the resulting clusters. Do
   they look meaningfully different? Which feature do you expect to
   dominate, and why?
3. An adjusted Rand index of `0.0` means "no better than random." What
   ARI would you expect if you clustered pure random noise (no real
   structure at all) against a hidden 3-category label? Why?

---

## Lesson 5: Teaching the Model — `churn_trainer.py`

**What you'll learn:** train/test splits, handling imbalanced classes,
and how scikit-learn `Pipeline`s bundle preprocessing with a model.

```python
def split_feature_table(
    feature_table: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split on the feature table - pulled out on its own
    so evaluation.py can fit a pipeline on the exact same split as here.
    """
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    y = feature_table[LABEL_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
```

**Concept: train/test split.** You never evaluate a model on the same
data it learned from — that would be like grading a student using the
exact questions they memorized answers to. `train_test_split` randomly
holds back `test_size=0.2` (20%) of the policies as a "test set" the model
never trains on, so the numbers in Lesson 6 reflect genuine
generalization, not memorization.

**Concept: stratification.** `stratify=y` makes sure the 20% test split
has roughly the same proportion of churned-vs-not policies as the full
dataset. Without it, a random split could accidentally put almost all the
churned examples in the training set (or the test set), badly skewing
what you can learn from either half. Since churn here is about 24% of
policies — not a 50/50 split — stratification matters more than it would
on a balanced dataset.

**Why is this split pulled into its own function?** So `evaluation.py`
(Lesson 6) and `explain.py` (Lesson 7) can call it too, and always get the
*exact same* train/test division as `churn_trainer.py` uses — same
`random_state`, same rows in each half. If each module split the data on
its own, they could silently drift out of sync, and a metric reported by
one module wouldn't describe the same held-out policies as another's.

```python
def build_pipeline(random_state: int = 42, classifier: ClassifierMixin | None = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), BEHAVIOR_FEATURE_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
```

**Concept: one-hot encoding.** `insurance_product` is text — "Auto",
"Home", "Life", and so on. Models need numbers, not text, and they
shouldn't assume "Life" is somehow numerically bigger than "Auto." One-hot
encoding solves this by creating one new 0/1 column per possible value
(`insurance_product_Auto`, `insurance_product_Home`, ...) — a policy gets
a `1` in exactly the column matching its actual product, `0` everywhere
else. `handle_unknown="ignore"` tells the encoder what to do if it ever
sees a category it didn't encounter during training (all zeros, rather
than crashing) — useful if a brand-new insurance product gets added later.

**Concept: `ColumnTransformer`.** This applies *different* preprocessing
to different columns in one step: standardize the numeric behavior
columns (same reasoning as Lesson 4), one-hot encode the two categorical
columns, and glue the results back together into one matrix.

**Concept: a scikit-learn `Pipeline`.** Bundling preprocessing and the
classifier into one object means there's exactly one thing to save, load,
and call `.predict_proba()` on later (Lesson 9) — no separate scaler or
encoder to accidentally get out of sync with the model at prediction
time.

**Concept: class imbalance, and `class_weight="balanced"`.** Only about
24% of policies in this dataset actually churned. A model trained without
any adjustment can get a deceptively good-looking accuracy just by
leaning toward always predicting "not churned" — it's right most of the
time by default, while being useless for the one thing that matters:
catching the churners. `class_weight="balanced"` tells the classifier to
penalize mistakes on the minority class (churned policies) more heavily
during training, counteracting that lazy shortcut.

```python
def build_random_forest_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)
```

**Logistic regression vs. random forest.** Logistic regression fits one
smooth, linear boundary between the two classes — simple, fast, and easy
to reason about. A random forest instead trains many decision trees
(`n_estimators=200` of them) on random subsets of the data and averages
their votes, which can capture more complex, non-linear patterns at the
cost of being harder to inspect directly. This project trained both and
compared them (Lesson 6) — the random forest ended up marginally more
accurate *and* pairs naturally with an exact explainability technique
(Lesson 7), which is why it's the one actually used for scoring.

```python
def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)
```

**Concept: model persistence.** Training takes real compute time; you
don't want to retrain from scratch every time you want to score a
policy. `joblib.dump`/`joblib.load` serialize the entire fitted pipeline
(preprocessing steps and all) to a file on disk (`models/churn_model.joblib`)
— save it once after training, load it instantly whenever scoring needs
it (Lesson 9).

A quick safety note, since `joblib.load` is built on Python's `pickle`
format: unpickling a file can execute arbitrary code, so you should only
ever load pickled files you trust. This project only ever loads files
that `save_pipeline()` itself wrote — never anything from an untrusted or
external source.

### Exercises

1. Change `test_size` to `0.5` and rerun `train_baseline_model()`. What
   happens to the reported metrics, and why might a 50/50 split be a
   worse choice than 20% for a dataset this size?
2. Remove `class_weight="balanced"` from `build_pipeline()`'s default
   `LogisticRegression` and compare `recall` before and after on the
   same data. What does that tell you about what `class_weight` is
   actually doing?
3. `OneHotEncoder(handle_unknown="ignore")` — what would happen instead
   if you used the default `handle_unknown="error"` and then tried to
   score a policy with an insurance product the model had never seen
   during training?

---

## Lesson 6: Grading the Model Honestly — `evaluation.py`

**What you'll learn:** why accuracy alone is misleading, what
precision/recall/F1/ROC-AUC actually measure, and how to pick a decision
threshold based on real business costs instead of a default.

**Why not just report accuracy?** Churn is about 24% of this dataset. A
model that predicts "not churned" for every single policy would already
be about 76% accurate — while being completely useless, since it would
never once flag a real churner. This module exists specifically to avoid
being fooled by that trap.

```python
def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
```

**The metrics, in plain English, using this project's real held-out
numbers (random forest, 74 held-out policies: 56 that didn't churn, 18
that did):**

- **Precision** — "of the policies I flagged as at-risk, how many
  actually churned?" At the default threshold, this model's precision is
  **1.000**: every policy it flagged really did churn. Zero wasted
  outreach.
- **Recall** — "of the policies that actually churned, how many did I
  catch?" Also **1.000** here: it caught every single one of the 18 real
  churners in the held-out set.
- **F1** — the harmonic mean of precision and recall, a single number
  that penalizes a model for being lopsided (very high precision but
  terrible recall, or vice versa). **1.000** here too, since both inputs
  were already perfect.
- **ROC-AUC** — roughly, "if I picked one churned policy and one
  non-churned policy at random, what's the chance the model scores the
  churned one higher?" 1.0 is perfect ranking, 0.5 is a coin flip.
  **1.000** here.
- **PR-AUC** — like ROC-AUC, but specifically better suited to imbalanced
  datasets like this one, since it doesn't get artificially inflated by
  a large pool of easy true negatives. **1.000** here as well.

(Real datasets are almost never this clean — this project's synthetic
data has an unusually strong, clear signal by design. The baseline
logistic regression, for comparison, scored precision 1.000 / recall
0.944 / ROC-AUC 0.997 — very good, but the random forest edged it out.)

```python
def confusion_counts(y_true, y_proba, threshold=DEFAULT_THRESHOLD) -> dict[str, int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
```

**Concept: confusion matrix.** The four raw counts every classification
metric above is built from: correctly-predicted negatives and positives
(true negative, true positive), and the two kinds of mistakes — a false
alarm (**false positive**: flagged as at-risk, but wouldn't have churned)
and a miss (**false negative**: churned, but the model didn't flag it).
`labels=[0, 1]` pins the matrix to always be 2×2, even in the edge case
where the model happens to predict only one class.

**Why these two mistakes matter differently.** A false positive costs a
wasted retention email or call. A false negative costs a customer who
leaves with zero warning. Those are *not* equally expensive mistakes in a
real business — and the next function is where this project explicitly
accounts for that instead of pretending they're symmetric.

```python
def pick_cost_weighted_threshold(
    y_true, y_proba, cost_fp=DEFAULT_COST_FALSE_POSITIVE, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, thresholds=None,
) -> tuple[float, pd.DataFrame]:
    sweep = threshold_sweep(y_true, y_proba, thresholds)
    sweep["expected_cost"] = sweep["false_positive"] * cost_fp + sweep["false_negative"] * cost_fn
    best_row = sweep.loc[sweep["expected_cost"].idxmin()]
    return float(best_row["threshold"]), sweep
```

**Concept: a decision threshold.** A classifier's raw output is a
*probability* — 0.62, say — not a yes/no answer. Turning that into a
decision means picking a cutoff: "flag anything at or above X%." The
obvious default is 50%, but 50% silently assumes both kinds of mistakes
cost exactly the same amount, which is rarely true. This function instead
sweeps across many candidate thresholds (`threshold_sweep()`, not shown in
full here — it just calls `compute_metrics()`/`confusion_counts()` at
each one, from 0.05 to 0.95), computes `expected_cost = false_positives ×
cost_fp + false_negatives × cost_fn` for each, and returns whichever
threshold minimizes that number.

By default `cost_fn` (5.0) is five times `cost_fp` (1.0) — reflecting
that missing a real churner is assumed to be several times more expensive
than one wasted outreach. On this project's data, that pushed the optimal
threshold down to **0.25** (lower than the default 0.5) — with false
negatives costing more, it's worth flagging a few extra borderline
policies (accepting some false positives) to avoid missing a real
churner.

**A note on calibration**, which the module also computes
(`calibration_table()`): a well-*calibrated* model's probabilities mean
what they say — among policies it scored around 70%, roughly 70% of them
should actually churn. This is a different question from ranking policies
correctly (which ROC-AUC measures) — a model can rank perfectly while
still being badly calibrated (e.g. every prediction is either 0.99 or
0.01, never anything in between). `calibration_table()` buckets
predictions into quantile bins and compares the average predicted
probability in each bucket to the actual observed churn rate there.

### Exercises

1. Change `DEFAULT_COST_FALSE_NEGATIVE` to `20.0` (missing a churner is
   now 20× as expensive as a false alarm) and rerun
   `pick_cost_weighted_threshold()`. Does the chosen threshold go up or
   down? Why does that direction make sense?
2. What would `pick_cost_weighted_threshold()` return if `cost_fp` and
   `cost_fn` were set equal to each other? Would you expect the result to
   land near the default 0.5 threshold?
3. Why is PR-AUC described above as "better suited to imbalanced
   datasets" than ROC-AUC? (Hint: think about what happens to each metric
   if you had 10,000 non-churned policies and only 10 churned ones.)

---

## Lesson 7: Explaining Predictions — `explain.py`

**What you'll learn:** why a probability alone isn't a satisfying answer,
and how SHAP breaks a prediction down into per-feature contributions.

**Why explainability matters here.** "This policy has a 73% churn risk"
doesn't tell a retention team what to *do*. They need to know *why* — is
it failed payments? A worsening trend? Something else? This module answers
that, in two flavors: per-policy ("why is *this* one flagged?") and
aggregate ("what behaviors matter across the whole book?").

```python
def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    X_transformed = transform_for_explaining(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    raw_values = explainer.shap_values(X_transformed)
    if isinstance(raw_values, list):
        churn_shap_values = raw_values[CHURN_CLASS_INDEX]
    else:
        churn_shap_values = raw_values[:, :, CHURN_CLASS_INDEX]
    return churn_shap_values, X_transformed
```

**Concept: SHAP values, in plain English.** Imagine the model has a
"baseline" prediction — what it would guess with zero information about a
specific policy, just the average churn rate across everyone. A SHAP
value is how much a single feature pushes that guess up or down for one
particular policy, in the units of probability. Add up the baseline plus
every feature's SHAP value, and you get back that policy's actual
predicted probability. A **positive** SHAP value pushed the prediction
*toward* churn; a **negative** one pushed it *away*.

**Concept: `TreeExplainer`, and why the model choice from Lesson 5
matters here.** SHAP has multiple ways to compute these contributions.
For tree-based models like a random forest, `TreeExplainer` can compute
*exact* SHAP values quickly, by walking the actual decision paths inside
the trees. A linear model like logistic regression would need a slower,
approximate, general-purpose method instead (`KernelExplainer`). This is
the second reason (alongside the raw accuracy numbers) the random forest
was chosen as the model this project actually explains and scores with.

**The messy bit: `isinstance(raw_values, list)`.** SHAP's own output
format for a binary classifier has changed across library versions —
older releases hand back a list of two arrays (one per class), current
ones (this project runs 0.52) hand back a single 3-dimensional array with
both classes stacked in the last axis. This `if` branch normalizes either
shape down to just the "churn" class's 2D array, so the rest of the
module doesn't need to care which SHAP version is installed. This is a
good real-world lesson: pin your dependency versions, but also write
defensively when a library's output shape isn't perfectly stable across
versions.

```python
def explain_policy(shap_values, X_transformed, row_position, top_n=5) -> pd.DataFrame:
    contributions = pd.DataFrame({
        "feature": X_transformed.columns,
        "value": X_transformed.iloc[row_position].to_numpy(),
        "shap_value": shap_values[row_position],
    })
    ranked = contributions.reindex(contributions["shap_value"].abs().sort_values(ascending=False).index)
    return ranked.head(top_n).reset_index(drop=True)
```

Ranking by the *absolute value* of the SHAP contribution (`.abs()`)
matters: a feature that strongly pushed the prediction toward "safe"
(large negative) is just as important to surface as one that pushed
strongly toward "risk" (large positive) — ranking by raw signed value
would bury the safe-pushing ones at the bottom even when they're the
biggest single factor in the score.

**A real worked example from this project's held-out test data:** the
riskiest held-out policy scored **100.0%** predicted risk. Its top three
SHAP contributors:

| feature | (standardized) value | SHAP value | direction |
|---|---|---|---|
| `tenure_days` | −3.34 | **+0.296** | pushes toward risk |
| `failed_invoice_ratio` | 1.43 | **+0.090** | pushes toward risk |
| `missed_invoice_ratio` | 0.74 | **+0.087** | pushes toward risk |

Compare that `tenure_days` row to our running example, `POL-1C1CDD7E`
(still Active, never cancelled), whose `tenure_days` SHAP value was
**−0.32** — pushing the *opposite* direction. Same feature, opposite
sign, because the underlying value is opposite: this held-out policy is a
Cancelled one with a short truncated tenure (recall Lesson 3's
`policy_static_features()` — a cancelled policy's tenure is measured to
its own `cancel_date`), while `POL-1C1CDD7E`'s tenure is the full 365-day
nominal term.

**An important interpretive catch this reveals:** among currently *open*
policies, virtually every one gets the same 365-day nominal
`tenure_days` value (every policy in this synthetic dataset has a fixed
one-year term), so `tenure_days`'s SHAP contribution ends up almost
constant across the whole active book — dominating the *aggregate*
importance ranking (`aggregate_feature_importance()`, below) without
actually differentiating one open policy's risk from another's. The
behavior features (`missed_invoice_ratio`, `failed_invoice_ratio`, and
friends) are where the real distinguishing signal lives for the policies
you'd actually want to act on.

```python
def aggregate_feature_importance(shap_values, X_transformed) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": X_transformed.columns,
        "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
    })
    return importance.sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)
```

Same idea as `explain_policy()`, but averaged across every row instead of
looking at just one: "on average, how much does each feature move the
needle, in either direction?" This project's current aggregate ranking
(74 held-out policies): `tenure_days` (0.311), `missed_invoice_ratio`
(0.098), `failed_invoice_ratio` (0.060), then a long tail of smaller
contributors.

### Exercises

1. For `POL-1C1CDD7E`, `failed_invoice_ratio`'s SHAP value was negative
   (pushing toward "safe"). Given its `failed_invoice_ratio` feature
   value is exactly `0.0`, does that direction make sense? Why?
2. `explain_policy()` defaults to `top_n=5`, but `churn_predict.py`
   (Lesson 9) calls it with `top_n=3`. What's the tradeoff of storing
   fewer top factors per policy — what real limitation could that cause?
   (This actually happens in this project — see if you can find where
   it's discussed in `docs/Technical.md`.)
3. Why does `aggregate_feature_importance()` use the *mean* of absolute
   SHAP values, rather than, say, the *maximum*? What would using the max
   emphasize instead?

---

## Lesson 8: Turning Insight into Action — `recommend.py`

**What you'll learn:** the simplest lesson in this course — a lookup
table — but a good example of encoding a business judgment call directly
in code.

```python
NON_ACTIONABLE_FEATURES = {"tenure_days"}

DRIVER_TO_ACTION: dict[str, str] = {
    "missed_invoice_ratio": "Proactive outreach before the next invoice is due - ...",
    "failed_invoice_ratio": "Prompt a payment-method update - ...",
    "late_invoice_ratio": "Offer a flexible due date or grace-period plan - ...",
    "avg_days_late": "Send payment reminders earlier in the billing cycle - ...",
    "max_days_late": "Escalate to a retention call - ...",
    "trend_failed_rate_delta": "Escalate to the retention team - ...",
    "trend_late_rate_delta": "Proactive check-in call - ...",
}

DEFAULT_ACTION = "Monitor - no single actionable behavioral driver stands out, ..."
```

**Why is this just a dictionary, not another model?** Not every problem
needs machine learning. Mapping "the top driver was failed payments" to
"suggest a payment-method update" is a straightforward business rule a
human can write down directly — there's no pattern here worth *learning*
from data, just a lookup. Recognizing when a rule-based approach is the
right (simpler, more auditable) tool instead of reaching for ML by
default is itself a useful engineering instinct.

**Why is `tenure_days` explicitly excluded?** Straight from Lesson 7:
telling a retention team "this policy is at risk because of its tenure"
isn't something they can act on — you can't change how long a policy has
existed. `NON_ACTIONABLE_FEATURES` exists specifically to keep this
outcome-adjacent feature from ever becoming a "suggested action," even
when it's the single largest SHAP contributor (which, per Lesson 7, it
usually is).

```python
def _actionable_risk_pushing_features(top_factors: list[dict]):
    """Shared filter behind recommend_action()/describe_factors()."""
    for factor in top_factors:
        feature = factor["feature"]
        value = factor["value"]
        if feature not in NON_ACTIONABLE_FEATURES and value > 0:
            yield feature


def recommend_action(top_factors: list[dict]) -> str:
    for feature in _actionable_risk_pushing_features(top_factors):
        if feature in DRIVER_TO_ACTION:
            return DRIVER_TO_ACTION[feature]
    return DEFAULT_ACTION
```

**Concept: a generator function.** `_actionable_risk_pushing_features` uses
`yield` instead of `return` — calling it doesn't run the function
immediately; it hands back an iterator that produces one qualifying
feature name at a time, on demand, as something (here, the `for` loop in
`recommend_action`) asks for the next one. `recommend_action()` walks that
stream and returns as soon as it finds the *first* feature name that's
also a key in `DRIVER_TO_ACTION` — a policy's top-ranked SHAP factor wins,
provided it's both actionable and actually pushing toward risk (`value > 0`
— a factor pulling the prediction the *other* way isn't something worth
intervening on).

**For `POL-1C1CDD7E`,** recall its top 3 SHAP factors from Lesson 7:
`tenure_days` (−0.32, excluded — non-actionable), `missed_invoice_ratio`
(+0.13, qualifies), `failed_invoice_ratio` (−0.05, would be excluded
anyway since it's negative). The first qualifying feature is
`missed_invoice_ratio`, so `recommend_action()` returns: *"Proactive
outreach before the next invoice is due — this policy has a history of
invoices going unpaid entirely."* That's a coherent story: this policy's
real problem is skipped bills, not payment failures, and the suggested
action matches.

```python
def describe_factors(top_factors: list[dict], max_items: int = 3) -> list[str]:
    descriptions: list[str] = []
    for feature in _actionable_risk_pushing_features(top_factors):
        if feature in DRIVER_TO_DESCRIPTION:
            descriptions.append(DRIVER_TO_DESCRIPTION[feature])
        if len(descriptions) >= max_items:
            break
    return descriptions
```

Same generator, same filter — but collecting up to `max_items` plain-language
descriptions instead of returning after the first match. This is what
powers a report's "why is this policy flagged?" bullet list, versus
`recommend_action()`'s single suggested action.

### Exercises

1. `recommend_action()` and `describe_factors()` both call the *same*
   generator function rather than duplicating the filtering logic. What
   would you have to remember to keep in sync if that filter were instead
   copy-pasted into both functions?
2. Add a fictional new feature, `"refund_request_count"`, to
   `DRIVER_TO_ACTION` with your own suggested action text. What happens
   if that feature name shows up in a policy's `top_factors` list but you
   forgot to add it to `DRIVER_TO_DESCRIPTION` too?
3. Why does `_actionable_risk_pushing_features()` check `value > 0`
   strictly, rather than `value >= 0`? Think about what a SHAP value of
   exactly `0` would mean.

---

## Lesson 9: Putting It Together — `churn_predict.py`

**What you'll learn:** orchestration — combining everything from Lessons
3–8 into one batch job — and a real bug this project found and fixed.

**The bug, first, because it motivates the most important function in
this file.** Early versions of this pipeline scored *every* policy,
regardless of whether it was still open. A Cancelled policy's outcome has
already happened — asking the model "what's this policy's churn risk?"
isn't a forward-looking prediction for one, it's just the model
recognizing its own training label reflected back through that policy's
behavior. In a real batch, this meant 164 of 175 policies flagged "at
risk" were already gone — not actionable, and actively misleading for
whoever reads the report.

```python
def filter_active_policies(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Only Active policies are worth scoring - a Cancelled/Expired one's
    outcome already happened, so a "churn risk" for it isn't a prediction.
    """
    return feature_table[feature_table["policy_status"] == ACTIVE_STATUS].reset_index(drop=True)
```

Simple code, important reasoning. Notice this is applied to the feature
table *before* scoring, clustering, or anything else runs — not as a
filter on the final report. **Where you apply a fix matters**: filtering
here means nothing downstream ever needs to remember to do it again.

```python
def score_all_policies(
    pipeline: Pipeline, feature_table: pd.DataFrame, top_n_factors: int = TOP_FACTORS_PER_POLICY
) -> pd.DataFrame:
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    churn_probability = pipeline.predict_proba(X)[:, CHURN_CLASS_INDEX]

    shap_values, X_transformed = compute_shap_values(pipeline, X)
    top_factors = [_row_top_factors(shap_values, X_transformed, i, top_n_factors) for i in range(len(X))]
    recommended_actions = [recommend_action(factors) for factors in top_factors]

    segmented = label_clusters_by_risk(cluster_policies(feature_table))

    scored_at = datetime.now(timezone.utc).replace(tzinfo=None)

    return pd.DataFrame({
        "policy_num": feature_table["policy_num"].to_numpy(),
        "scored_at": scored_at,
        "churn_probability": churn_probability.round(4),
        "segment_label": segmented["segment_label"].to_numpy(),
        "top_factors": top_factors,
        "recommended_action": recommended_actions,
    })
```

Watch how much this one function pulls together, all on the *filtered*
Active-only feature table: `pipeline.predict_proba()` from Lesson 5's
persisted model, `compute_shap_values()`/`explain_policy()`-via-`_row_top_factors()`
from Lesson 7, `cluster_policies()`/`label_clusters_by_risk()` from
Lesson 4 (reclustering fresh on just the active book, not reusing any
saved cluster centers), and `recommend_action()` from Lesson 8. This is
the payoff of every earlier module being small and independently
reusable: none of that logic had to be rewritten here.

**Why every row shares the same `scored_at` timestamp**, computed once
before the loop rather than once per row: this batch of scores is
meant to be read back together as "the state of the book as of this one
moment." `tbl_policy_risk_score`'s primary key is `(policy_num, scored_at)`
together, not `policy_num` alone — so rerunning this script tomorrow adds
a *new* batch of rows rather than overwriting today's, letting a risk
trend accumulate over time instead of only ever showing the latest
number.

```python
def compute_batch_feature_importance(pipeline: Pipeline, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SHAP feature importance across the whole batch - a separate
    call from score_all_policies() since the aggregate view needs the full
    per-feature SHAP matrix, not just each policy's top few factors.
    """
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    shap_values, X_transformed = compute_shap_values(pipeline, X)
    return aggregate_feature_importance(shap_values, X_transformed)
```

Why compute SHAP values *twice* — once inside `score_all_policies()`,
again here? Because `score_all_policies()` only keeps each policy's
*top 3* factors (to keep the stored JSON small), which silently drops
every smaller contributor. A true aggregate importance ranking needs the
*full* SHAP matrix (every feature's contribution for every policy), not
just the parts that happened to make each individual policy's top 3.

```python
def run_batch_scoring(engine: Engine | None = None) -> pd.DataFrame:
    """Full pipeline: load the persisted model, score every Active policy, write results to postgres + models/."""
    engine = engine or get_engine()
    pipeline = load_pipeline()
    feature_table = filter_active_policies(build_feature_table(engine))

    scores = score_all_policies(pipeline, feature_table)
    inserted = write_risk_scores(engine, scores)

    importance = compute_batch_feature_importance(pipeline, feature_table)
    importance_path = save_feature_importance(importance)
    ...
    return scores
```

This is the entry point (`python -m churn.churn_predict` runs it). Read
it as a plain sentence: load the saved model, build the feature table,
keep only Active policies, score them, save the scores to Postgres, and
save the aggregate importance to a CSV. Every piece it calls was built
and explained in an earlier lesson.

### Exercises

1. `filter_active_policies()` is applied to `build_feature_table(engine)`'s
   *output* — after all four raw tables have already been read and
   joined. Could you instead filter earlier, say directly in the SQL
   query in `features.py`? What would you gain or lose by doing that?
2. What would go wrong if `score_all_policies()` computed a fresh
   `scored_at` timestamp inside the list comprehension that builds
   `top_factors`, once per policy, instead of once before the DataFrame
   is built?
3. `compute_batch_feature_importance()` recomputes SHAP values that
   `score_all_policies()` already computed once. For a much larger book
   of policies (say, a million), what would you consider changing about
   this design to avoid the duplicate work?

---

## Lesson 10: Delivering the Result — `report.py`

**What you'll learn:** the basics of generating a real Excel file from
Python — workbooks, worksheets, cell styling, and native charts.

This module deliberately does **no** machine learning at all — it only
reads what `churn_predict.py` already computed and formats it.

```python
def load_latest_risk_scores(engine: Engine | None = None) -> pd.DataFrame:
    return pd.read_sql(LATEST_RISK_SCORES_QUERY, engine or get_engine())
```

`LATEST_RISK_SCORES_QUERY` is a plain SQL string that joins
`tbl_policy_risk_score` to `tbl_policies` (to pull in product/status
context) and filters to only the most recent `scored_at` batch — a plain
subquery, `WHERE scored_at = (SELECT MAX(scored_at) ...)`.

```python
def build_policy_detail_table(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per policy, business-facing columns only - no raw SHAP values or feature names."""
    detail = scores.copy()
    detail["why_flagged"] = detail["top_factors"].apply(
        lambda factors: "; ".join(describe_factors(factors)) or "-"
    )
    ordered_columns = list(DETAIL_COLUMN_RENAME)
    return detail[ordered_columns].rename(columns=DETAIL_COLUMN_RENAME)
```

This is a deliberate translation step: raw column names like
`churn_probability` become reader-friendly headers like "Churn Risk," and
the SHAP-derived `top_factors` JSON becomes a plain sentence via
`describe_factors()` from Lesson 8 — nobody reading this report in Excel
needs to know what SHAP is.

**Now the Excel-specific part.** `openpyxl` is the library used here to
build a real `.xlsx` file — not just write a CSV and hope Excel opens it
nicely, but construct an actual workbook object in memory.

```python
def write_policy_detail_sheet(workbook: Workbook, detail: pd.DataFrame) -> Worksheet:
    sheet = workbook.active
    sheet.title = "Policy Detail"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="31333F")
    sheet.append(list(detail.columns))
    for cell in sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center")

    for _, row in detail.iterrows():
        sheet.append(list(row))

    segment_col = list(detail.columns).index("Segment") + 1
    risk_col = list(detail.columns).index("Churn Risk") + 1
    for row_idx in range(2, sheet.max_row + 1):
        segment = sheet.cell(row=row_idx, column=segment_col).value
        fill = SEGMENT_FILL.get(segment)
        if fill:
            sheet.cell(row=row_idx, column=segment_col).fill = fill
        sheet.cell(row=row_idx, column=risk_col).number_format = "0%"

    for i, column_name in enumerate(detail.columns, start=1):
        sheet.column_dimensions[get_column_letter(i)].width = DETAIL_COLUMN_WIDTHS.get(column_name, 16)

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    return sheet
```

**Concept: `Workbook`/`Worksheet`.** A `Workbook` is the whole `.xlsx`
file; a `Worksheet` is one tab inside it. Every new workbook starts with
one sheet (`workbook.active`), which this function renames to "Policy
Detail" and fills in row by row with `sheet.append(...)`.

**Cell-level styling.** `Font`, `PatternFill`, and `Alignment` are
openpyxl objects that describe how a cell should look — bold white text
on a dark background for the header row, for instance. `sheet[1]` means
"every cell in row 1." Notice this is done with plain loops over rows and
columns — there's no higher-level "make this a styled table" shortcut
being used here, just direct cell manipulation, which is normal for
openpyxl.

**Conditional formatting by hand.** `SEGMENT_FILL` is a dict mapping
segment names to colors (light red/amber/green). The loop walks every
data row, looks up that row's segment in the dict, and applies the
matching fill — this is what makes "High risk" rows visibly red in the
final spreadsheet. `number_format = "0%"` tells Excel to *display* the
underlying float (like `0.205`) as a percentage (`21%`) without changing
the stored value at all.

**Two features that make the sheet actually usable, not just
pretty:** `sheet.freeze_panes = "A2"` keeps the header row visible while
scrolling, and `sheet.auto_filter.ref = sheet.dimensions` turns on
Excel's built-in column filter/sort controls across the whole populated
range — letting the person reading the report filter to just "High risk"
themselves, rather than needing a fixed dropdown built ahead of time.

```python
def _add_bar_chart(sheet: Worksheet, title: str, header_row: int, row_count: int, anchor_row: int) -> None:
    chart = BarChart()
    chart.title = title
    chart.y_axis.title = "Policies"
    data = Reference(sheet, min_col=2, min_row=header_row, max_row=header_row + row_count)
    categories = Reference(sheet, min_col=1, min_row=header_row + 1, max_row=header_row + row_count)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    chart.width, chart.height = 16, 8
    sheet.add_chart(chart, f"D{anchor_row}")
```

**Concept: a `Reference`.** Rather than handing a chart raw numbers,
openpyxl charts point at a *range of cells* — `Reference(sheet, min_col=2,
min_row=header_row, max_row=header_row + row_count)` means "column B,
from this row down to that row." This means the chart stays a **real,
live Excel chart** — if you edited the underlying numbers by hand in
Excel, the chart would update, exactly like any chart you built manually
in the app. That's meaningfully different from generating a picture of a
chart and pasting it in.

```python
def build_report(scores: pd.DataFrame, path: Path | None = None) -> Path:
    if scores.empty:
        raise ValueError("No scored policies to report on - run `python -m churn.churn_predict` first.")
    path = path or default_report_path()

    detail = build_policy_detail_table(scores)
    segment_counts, action_counts = build_summary_tables(scores)

    workbook = Workbook()
    write_policy_detail_sheet(workbook, detail)
    write_summary_sheet(workbook, scores, segment_counts, action_counts)

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path
```

```python
def default_report_path(today: date | None = None) -> Path:
    """models/risk_report_YYYYMMDD.xlsx for the given (or current) date."""
    today = today or date.today()
    return MODEL_DIR / f"risk_report_{today:%Y%m%d}.xlsx"
```

Notice the empty-input check right at the top of `build_report()` — a
deliberate, clear error ("run `churn_predict` first") rather than
silently producing an empty, confusing spreadsheet. And notice
`default_report_path()` takes an optional `today` parameter instead of
just calling `date.today()` directly inside itself — that's what makes it
possible to test this function's date-formatting logic with a fixed,
known date instead of whatever day happens to be "today" when the test
runs (see `tests/test_report.py`).

### Exercises

1. Add a new column to `DETAIL_COLUMN_RENAME`/`DETAIL_COLUMN_WIDTHS` for
   `payment_frequency` positioned right after "Product," and adjust
   `build_policy_detail_table()`'s `ordered_columns` accordingly.
2. `write_policy_detail_sheet()` loops over every row with
   `range(2, sheet.max_row + 1)` to apply conditional formatting — why
   does it start at `2`, not `1`?
3. What would happen if two policies had the exact same `segment_label`
   value but with different capitalization (`"high risk"` vs.
   `"High risk"`)? Would `SEGMENT_FILL.get(segment)` still color both
   rows correctly? Why or why not?

---

## Lesson 11 — Following One Policy Through Everything

Let's put it all together. Here is `POL-1C1CDD7E`'s entire journey,
start to finish, one lesson at a time.

**1–2. Raw data.** A Home policy, Quarterly payments, 5 invoices, loaded
into Postgres by `seed_db.py` from a synthetic CSV. No failed payments —
just 2 missed invoices and 2 paid late.

**3. Features (`features.py`).** Collapsed to one row:
`missed_invoice_ratio=0.40`, `failed_invoice_ratio=0.0`,
`late_invoice_ratio=0.40`, `avg_days_late=14.67`, `max_days_late=29`,
`trend_failed_rate_delta=0.0`, `trend_late_rate_delta=0.167`,
`tenure_days=365` (the full nominal term — it's Active), `is_churned=0`.

**4. Segmentation (`segmentation.py`).** Clustered alongside every other
currently-Active policy, purely on those behavior numbers. It lands in
the **"Medium risk"** cluster.

**5. Training (`churn_trainer.py`).** This specific policy wasn't
necessarily in the model's test set — but whichever split it fell in, the
random forest it was trained against learned to recognize patterns like
"moderate missed-payment ratio, no failures, full tenure" from thousands
of trees built across the whole labeled dataset (churned and not).

**6. Evaluation (`evaluation.py`).** Doesn't touch this policy directly —
it grades the *model as a whole* on a held-out slice. But it's what
determined the cost-weighted decision threshold (0.25) that will
eventually decide whether this policy's own score counts as "flag it" or
not.

**7. Explainability (`explain.py`).** Run through `compute_shap_values()`,
this policy's prediction breaks down as: `tenure_days` **−0.32** (pushes
safe — full-term tenure looks nothing like a cancelled policy's),
`missed_invoice_ratio` **+0.13** (pushes risk — real missed-payment
history), `failed_invoice_ratio` **−0.05** (pushes safe — correctly,
since it's genuinely 0).

**8. Recommendation (`recommend.py`).** `tenure_days` is skipped
(non-actionable). `missed_invoice_ratio` is the first qualifying,
risk-pushing feature → *"Proactive outreach before the next invoice is
due — this policy has a history of invoices going unpaid entirely."*

**9. Batch scoring (`churn_predict.py`).** Because this policy's
`policy_status` is `Active`, it survives `filter_active_policies()` and
gets scored: **churn_probability = 0.205 (20.5%)**. Since 20.5% is below
the cost-weighted threshold of 0.25, this specific policy would *not* be
flagged as urgent under that threshold — though its "Medium risk" segment
label (a separate, relative signal from clustering) still surfaces it in
the report. One row gets written to `tbl_policy_risk_score`, with
`top_factors` as a JSON array and the recommended action as text.

**10. The report (`report.py`).** That row gets joined with
`tbl_policies` for its product/status, translated into a "Why Flagged"
sentence ("History of missed payments"), color-coded amber for "Medium
risk," and lands as one row in `risk_report_YYYYMMDD.xlsx` — a plain
spreadsheet row a retention analyst can read without ever knowing what
SHAP, KMeans, or a random forest are.

That's the whole pipeline, traced through one real row of data.

---

## Glossary

**Adjusted Rand index (ARI)** — a score from roughly 0 (no better than
random) to 1 (perfect match) summarizing how well two different groupings
of the same items agree with each other.

**Calibration** — whether a model's predicted probabilities match
reality (a 70%-predicted bucket should actually see about 70% of that
group experience the outcome).

**Class imbalance** — when one outcome (here, churn) is much rarer than
the other in your dataset, which can bias a naively-trained model toward
just predicting the common outcome every time.

**ColumnTransformer** (scikit-learn) — applies different preprocessing
steps to different columns of the same table in one combined step.

**Confusion matrix** — the 2×2 (for binary classification) breakdown of
true positives, true negatives, false positives, and false negatives.

**Engine** (SQLAlchemy) — a reusable factory object that manages a pool
of database connections, rather than one single open connection.

**F1 score** — the harmonic mean of precision and recall; a single
number that penalizes a model for being very good at one and very bad at
the other.

**Feature** — one measurable input a model learns from (e.g.
`missed_invoice_ratio`). **Feature engineering** is the process of
turning raw data into useful features.

**Idempotent** — an operation that produces the same end result no
matter how many times you run it.

**Joblib** — a Python library for saving ("serializing") objects like a
fitted model to disk and loading them back later.

**KMeans** — an unsupervised clustering algorithm that groups data points
by repeatedly assigning them to the nearest of *k* center points and
recomputing those centers.

**Label** — the "correct answer" a supervised model learns to predict
(here, `is_churned`). Unsupervised methods like KMeans never see the
label.

**One-hot encoding** — representing a text category as several 0/1
columns, one per possible value, so a model can use it as numeric input
without implying a false ordering.

**Pipeline** (scikit-learn) — bundles preprocessing steps and a model
into one object that can be fit, saved, and used to predict as a single
unit.

**Precision** — of everything you flagged as positive, what fraction
actually was.

**PR-AUC** — the area under the precision-recall curve; like ROC-AUC but
more informative on imbalanced datasets.

**Recall** — of everything that actually was positive, what fraction you
successfully flagged.

**ROC-AUC** — the probability that a model ranks a randomly-chosen
positive example higher than a randomly-chosen negative one; 1.0 is
perfect, 0.5 is random guessing.

**SHAP value** — how much one feature contributed (positively or
negatively) to moving one specific prediction away from the model's
baseline average.

**Standardization** (z-scoring) — rescaling a numeric column to have
mean 0 and standard deviation 1, so columns on very different scales can
be compared fairly.

**Stratified split** — a train/test split that preserves the same class
proportions in both halves as in the full dataset.

**Threshold** — the probability cutoff above which a prediction counts
as "positive." Choosing it well (rather than defaulting to 0.5) is a
decision about the relative cost of the two kinds of mistakes.

**Unsupervised learning** — finding structure in data without being told
the "right answer" for any example.

**Upsert** — a database write that inserts a new row, or updates/skips
it if a matching one already exists.
