"""
Quick EDA pass over the churn data sitting in postgres.

Note: the DB has no "profile" column - good/at_risk/churned was just an
internal concept in the synthetic data generator, never persisted. So the
closest real stand-in for "risk segment" here is policy_status, and we
compare payment behavior across those groups. That's basically the
signal a churn model needs to pick up on.

Usage:
    python notebooks/eda.py
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from churn.churn_db import get_engine


def load_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    # dataset's small, just pull everything into memory - keeps this simple
    table_names = ["tbl_customer", "tbl_policies", "tbl_invoice", "tbl_payhistory"]
    return {name: pd.read_sql_table(name, engine) for name in table_names}


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def class_balance(policies: pd.DataFrame) -> None:
    print_header("Policy status (churn label) balance")
    counts = policies["policy_status"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    for status, count in counts.items():
        print(f"  {status:<12} {count:>5}  ({pct[status]}%)")


def payment_status_by_policy_status(
    policies: pd.DataFrame, invoices: pd.DataFrame, payhistory: pd.DataFrame
) -> None:
    print_header("Payment status distribution by policy status (row %)")
    # walk payhistory -> invoice -> policy so we know which policy_status each payment belongs to
    merged = payhistory.merge(invoices[["invoice_num", "policy_num"]], on="invoice_num")
    merged = merged.merge(policies[["policy_num", "policy_status"]], on="policy_num")

    table = pd.crosstab(
        merged["policy_status"], merged["pay_transaction_status"], normalize="index"
    ) * 100
    print(table.round(1))


def missed_invoice_ratio(
    policies: pd.DataFrame, invoices: pd.DataFrame, payhistory: pd.DataFrame
) -> None:
    print_header("Missed-invoice ratio by policy status (invoices with zero payments)")
    invoices_with_policy = invoices.merge(policies[["policy_num", "policy_status"]], on="policy_num")
    paid_invoice_nums = set(payhistory["invoice_num"])
    invoices_with_policy["has_payment"] = invoices_with_policy["invoice_num"].isin(paid_invoice_nums)

    summary = invoices_with_policy.groupby("policy_status")["has_payment"].agg(total="count", paid="sum")
    summary["missed"] = summary["total"] - summary["paid"]
    summary["missed_pct"] = (summary["missed"] / summary["total"] * 100).round(1)
    print(summary)


def policy_and_product_summary(policies: pd.DataFrame) -> None:
    print_header("Policies per customer")
    per_cust = policies.groupby("customer_id").size()
    print(f"  mean: {per_cust.mean():.2f}   min: {per_cust.min()}   max: {per_cust.max()}")

    print_header("Insurance product distribution")
    print(policies["insurance_product"].value_counts())

    print_header("Payment frequency distribution")
    print(policies["payment_frequency"].value_counts())


def run_eda() -> None:
    engine = get_engine()
    data = load_tables(engine)

    print(
        f"customers={len(data['tbl_customer'])}  policies={len(data['tbl_policies'])}  "
        f"invoices={len(data['tbl_invoice'])}  payhistory={len(data['tbl_payhistory'])}"
    )

    class_balance(data["tbl_policies"])
    payment_status_by_policy_status(data["tbl_policies"], data["tbl_invoice"], data["tbl_payhistory"])
    missed_invoice_ratio(data["tbl_policies"], data["tbl_invoice"], data["tbl_payhistory"])
    policy_and_product_summary(data["tbl_policies"])


if __name__ == "__main__":
    run_eda()
