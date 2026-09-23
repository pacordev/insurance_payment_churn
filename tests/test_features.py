"""
Tests for the per-policy feature table in features.py. All built from plain
DataFrames (no live db needed), same approach as test_seed_db.py.
"""

from __future__ import annotations

import pandas as pd

from sqlalchemy.sql.elements import quoted_name

from churn.features import (
    _trend_delta,
    aggregate_policy_features,
    policy_static_features,
    stringify_columns,
    summarize_payhistory_by_invoice,
)


def _invoices(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["invoice_start_date"] = pd.to_datetime(df["invoice_start_date"])
    return df


def test_invoice_with_no_payhistory_row_is_missed() -> None:
    invoices = _invoices(
        [{"invoice_num": "INV-1", "policy_num": "POL-1", "installment_num": 1,
          "invoice_start_date": "2024-01-01", "days_arrears_allowed": 30}]
    )
    # empty but with real dtypes, since a live read from postgres would never hand us
    # an all-object frame here - this is just matching that shape for the test
    payhistory = pd.DataFrame(
        {
            "invoice_num": pd.Series(dtype="object"),
            "pay_transaction_date": pd.Series(dtype="datetime64[ns]"),
            "pay_transaction_status": pd.Series(dtype="object"),
        }
    )

    summary = summarize_payhistory_by_invoice(invoices, payhistory)

    row = summary.iloc[0]
    assert row["was_missed"]
    assert not row["had_failed_attempt"]
    assert not row["was_paid_late"]


def test_failed_then_successful_retry_still_counts_as_failed_not_missed() -> None:
    invoices = _invoices(
        [{"invoice_num": "INV-1", "policy_num": "POL-1", "installment_num": 1,
          "invoice_start_date": "2024-01-01", "days_arrears_allowed": 30}]
    )
    payhistory = pd.DataFrame(
        [
            {"invoice_num": "INV-1", "pay_transaction_date": pd.Timestamp("2024-01-05"),
             "pay_transaction_status": "Failed"},
            {"invoice_num": "INV-1", "pay_transaction_date": pd.Timestamp("2024-01-10"),
             "pay_transaction_status": "Paid"},
        ]
    )

    summary = summarize_payhistory_by_invoice(invoices, payhistory)

    row = summary.iloc[0]
    assert not row["was_missed"]
    assert row["had_failed_attempt"]
    assert row["failed_attempts"] == 1


def test_days_late_only_counts_time_past_the_arrears_window() -> None:
    invoices = _invoices(
        [{"invoice_num": "INV-ON-TIME", "policy_num": "POL-1", "installment_num": 1,
          "invoice_start_date": "2024-01-01", "days_arrears_allowed": 30},
         {"invoice_num": "INV-LATE", "policy_num": "POL-1", "installment_num": 2,
          "invoice_start_date": "2024-01-01", "days_arrears_allowed": 30}]
    )
    payhistory = pd.DataFrame(
        [
            # paid on day 5 - well inside the 30 day arrears window, so 0 days late
            {"invoice_num": "INV-ON-TIME", "pay_transaction_date": pd.Timestamp("2024-01-06"),
             "pay_transaction_status": "Paid"},
            # paid on day 40 - 10 days past the 30 day arrears window
            {"invoice_num": "INV-LATE", "pay_transaction_date": pd.Timestamp("2024-02-10"),
             "pay_transaction_status": "Paid Late"},
        ]
    )

    summary = summarize_payhistory_by_invoice(invoices, payhistory)
    by_invoice = summary.set_index("invoice_num")

    assert by_invoice.loc["INV-ON-TIME", "max_days_late"] == 0
    assert by_invoice.loc["INV-LATE", "max_days_late"] == 10


def test_trend_delta_picks_up_more_failures_recently() -> None:
    # early half all good (0), recent half all failed (1) -> delta should be positive and large
    flags = pd.Series([0.0, 0.0, 1.0, 1.0])

    delta = _trend_delta(flags)

    assert delta == 1.0


def test_trend_delta_is_zero_with_fewer_than_two_invoices() -> None:
    assert _trend_delta(pd.Series([1.0])) == 0.0
    assert _trend_delta(pd.Series([], dtype=float)) == 0.0


def test_aggregate_policy_features_ratios_and_counts() -> None:
    invoices = _invoices(
        [
            {"invoice_num": "INV-1", "policy_num": "POL-1", "installment_num": 1,
             "invoice_start_date": "2024-01-01", "days_arrears_allowed": 30},
            {"invoice_num": "INV-2", "policy_num": "POL-1", "installment_num": 2,
             "invoice_start_date": "2024-02-01", "days_arrears_allowed": 30},
        ]
    )
    # INV-1 paid on time, INV-2 missed entirely
    payhistory = pd.DataFrame(
        [
            {"invoice_num": "INV-1", "pay_transaction_date": pd.Timestamp("2024-01-05"),
             "pay_transaction_status": "Paid"},
        ]
    )

    summary = summarize_payhistory_by_invoice(invoices, payhistory)
    features = aggregate_policy_features(summary).set_index("policy_num")

    row = features.loc["POL-1"]
    assert row["total_invoices"] == 2
    assert row["missed_invoice_count"] == 1
    assert row["missed_invoice_ratio"] == 0.5
    assert row["failed_invoice_count"] == 0


def test_tenure_uses_cancel_date_for_cancelled_policies() -> None:
    policies = pd.DataFrame(
        [
            {"policy_num": "POL-CANCELLED", "customer_id": "CUST-1", "insurance_product": "Auto",
             "payment_frequency": "Monthly", "policy_status": "Cancelled",
             "effective_start_date": pd.Timestamp("2024-01-01"),
             "effective_end_date": pd.Timestamp("2025-01-01"),
             "cancel_date": pd.Timestamp("2024-04-01")},
            {"policy_num": "POL-ACTIVE", "customer_id": "CUST-2", "insurance_product": "Auto",
             "payment_frequency": "Monthly", "policy_status": "Active",
             "effective_start_date": pd.Timestamp("2024-01-01"),
             "effective_end_date": pd.Timestamp("2025-01-01"),
             "cancel_date": None},
        ]
    )

    features = policy_static_features(policies).set_index("policy_num")

    # cancelled: Jan 1 -> Apr 1 = 91 days, not the full year to effective_end_date
    assert features.loc["POL-CANCELLED", "tenure_days"] == 91
    assert features.loc["POL-CANCELLED", "is_churned"] == 1
    # active: runs the full effective_start -> effective_end span
    assert features.loc["POL-ACTIVE", "tenure_days"] == 366
    assert features.loc["POL-ACTIVE", "is_churned"] == 0


def test_stringify_columns_normalizes_quoted_name_to_plain_str() -> None:
    # regression test: pd.read_sql_table columns come back as quoted_name
    # (a str subclass) - .astype(str) alone is a no-op on those since
    # quoted_name already satisfies pandas' "is this a string" check, so
    # this must produce genuine str instances, not just "string-like" ones
    df = pd.DataFrame({quoted_name("policy_num", None): ["POL-1"], "tenure_days": [10]})
    assert {type(c) for c in df.columns} == {quoted_name, str}  # the mix that breaks sklearn

    result = stringify_columns(df)

    assert {type(c) for c in result.columns} == {str}
