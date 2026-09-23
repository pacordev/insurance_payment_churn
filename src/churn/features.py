"""
Per-policy feature table for the churn model.

Turns the four raw tables (customer -> policies -> invoice -> payhistory)
into one row per policy with aggregated payment-behavior features. This is
the thing churn_trainer.py will eventually train on.

Usage:
    python -m churn.features
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy.engine import Engine

from churn.churn_db import get_engine

RAW_TABLE_NAMES = ["tbl_customer", "tbl_policies", "tbl_invoice", "tbl_payhistory"]

# payment-behavior columns build_feature_table() produces - shared by
# segmentation.py and churn_trainer.py so both cluster/train on the exact
# same numeric feature set instead of two hand-maintained copies of this list.
# Ratios/rates rather than raw counts (redundant with total_invoices), and
# never is_churned/policy_status - that's the label, not a feature.
BEHAVIOR_FEATURE_COLUMNS = [
    "missed_invoice_ratio",
    "failed_invoice_ratio",
    "late_invoice_ratio",
    "avg_days_late",
    "max_days_late",
    "trend_failed_rate_delta",
    "trend_late_rate_delta",
    "tenure_days",
]


def load_raw_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    """Pull all four raw tables into memory - dataset's small, keeps this simple."""
    return {name: pd.read_sql_table(name, engine) for name in RAW_TABLE_NAMES}


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
    # no successfully-timed payment at all (everything missed/failed) leaves these NaN -
    # 0.0 ("no lateness observed") is the sensible default rather than dropping the policy
    features[["avg_days_late", "max_days_late"]] = features[["avg_days_late", "max_days_late"]].fillna(0.0)
    return features.reset_index()


def policy_static_features(policies: pd.DataFrame) -> pd.DataFrame:
    """Policy-level fields that don't need any payhistory rollup: tenure + the label."""
    df = policies.copy()
    # cancelled policies "end" at cancel_date, everyone else at their normal effective_end_date
    end_date = df["cancel_date"].where(df["policy_status"] == "Cancelled", df["effective_end_date"])
    df["tenure_days"] = (pd.to_datetime(end_date) - pd.to_datetime(df["effective_start_date"])).dt.days
    df["is_churned"] = (df["policy_status"] == "Cancelled").astype(int)
    return df[
        [
            "policy_num",
            "customer_id",
            "insurance_product",
            "payment_frequency",
            "policy_status",
            "tenure_days",
            "is_churned",
        ]
    ]


def stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    pd.read_sql_table hands back column names as sqlalchemy's `quoted_name`
    (a str subclass) for columns sourced straight from postgres, while
    pandas-computed columns end up plain str - a mix scikit-learn now
    rejects outright when a DataFrame with both is fed into a pipeline.
    .astype(str) is a no-op here since quoted_name already passes pandas'
    "is this a string" check, so rebuild the index with real str() calls
    instead to force plain str instances.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def build_feature_table(engine: Engine | None = None) -> pd.DataFrame:
    """Build the full one-row-per-policy feature table straight from postgres."""
    engine = engine or get_engine()
    tables = load_raw_tables(engine)

    invoice_summary = summarize_payhistory_by_invoice(tables["tbl_invoice"], tables["tbl_payhistory"])
    payment_features = aggregate_policy_features(invoice_summary)
    static = policy_static_features(tables["tbl_policies"])

    combined = static.merge(payment_features, on="policy_num", how="left")
    # a policy with literally zero invoices (shouldn't happen, but just in case) gets
    # all-zero payment behavior rather than NaNs leaking into the model later
    count_and_ratio_cols = [c for c in payment_features.columns if c != "policy_num"]
    combined[count_and_ratio_cols] = combined[count_and_ratio_cols].fillna(0.0)
    return stringify_columns(combined)


if __name__ == "__main__":
    table = build_feature_table()
    print(f"feature table: {table.shape[0]} policies x {table.shape[1]} columns")
    print(table.dtypes)
    print(table.head())
