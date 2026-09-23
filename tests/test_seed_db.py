"""
Regression tests for seed_db.py's date handling.

Mainly guards against the NaT-leak bug: df.where(pd.notnull(df), None)
silently keeps NaT on a datetime64 column instead of nulling it, which
corrupted Postgres with a garbage date (48113-11-21) for every
non-cancelled policy. See churn_plan.md > Decisions for the write-up.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from churn.seed_db import clean_date_columns


def test_missing_date_becomes_none_not_nat() -> None:
    df = pd.DataFrame(
        {
            "policy_num": ["POL-1", "POL-2"],
            "cancel_date": pd.to_datetime(["2024-01-15", None]),
        }
    )

    cleaned = clean_date_columns(df, ["cancel_date"])

    assert cleaned["cancel_date"].iloc[0] == date(2024, 1, 15)
    assert cleaned["cancel_date"].iloc[1] is None


def test_missing_date_never_leaks_nat_sentinel_as_garbage_date() -> None:
    # this is the exact regression case: a blank cancel_date must never come
    # out as some huge/garbage year from pandas' internal NaT sentinel
    df = pd.DataFrame({"cancel_date": pd.to_datetime([None])})

    cleaned = clean_date_columns(df, ["cancel_date"])

    value = cleaned["cancel_date"].iloc[0]
    assert value is None
    assert not isinstance(value, pd.Timestamp)


def test_present_dates_are_plain_python_date_objects() -> None:
    df = pd.DataFrame({"invoice_start_date": pd.to_datetime(["2023-05-01"])})

    cleaned = clean_date_columns(df, ["invoice_start_date"])

    value = cleaned["invoice_start_date"].iloc[0]
    assert value == date(2023, 5, 1)
    assert type(value) is date
