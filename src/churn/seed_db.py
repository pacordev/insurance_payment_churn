"""
Seed script: loads the synthetic CSVs (syntethic_data/) into postgres.

This is NOT meant to be run only once. Since this is a testing/learning
ML project, the plan is to occasionally generate more synthetic data
(different seed / customer count) and just rerun this script to merge
the new rows in. Row IDs are UUID-based so fresh batches won't collide
with what's already loaded - the "ON CONFLICT DO NOTHING" below is just
a safety net in case you reload the exact same CSVs twice.

Usage:
    python -m churn.seed_db
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from churn.churn_db import get_engine

# project_root/syntethic_data
DATA_DIR = Path(__file__).resolve().parents[2] / "syntethic_data"

# load order matters because of the FK chain: customer -> policies -> invoice -> payhistory
# each entry is (table_name, columns that need to be parsed as dates)
TABLES_IN_LOAD_ORDER: list[tuple[str, list[str]]] = [
    ("tbl_customer", []),
    ("tbl_policies", ["effective_start_date", "effective_end_date", "cancel_date"]),
    ("tbl_invoice", ["invoice_start_date", "invoice_end_date"]),
    ("tbl_payhistory", ["pay_transaction_date"]),
]


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


def load_csv(table_name: str, date_cols: list[str]) -> pd.DataFrame:
    """Read one table's csv and turn empty/blank dates into real None values."""
    csv_path = DATA_DIR / f"{table_name}.csv"
    df = pd.read_csv(csv_path, parse_dates=date_cols)
    return clean_date_columns(df, date_cols)


def upsert_dataframe(engine: Engine, table_name: str, df: pd.DataFrame) -> int:
    """Insert every row in df, skipping ones that already exist (same primary key)."""
    if df.empty:
        return 0

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    rows = df.to_dict(orient="records")
    stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    # RETURNING only gives back the rows that were actually inserted, so
    # counting them is the reliable way to know how many were new
    # (plain rowcount isn't trustworthy for a multi-row INSERT ... ON CONFLICT)
    stmt = stmt.returning(table.primary_key.columns.values()[0])

    with engine.begin() as conn:
        result = conn.execute(stmt)
        inserted_ids = result.fetchall()

    return len(inserted_ids)


def seed_all(engine: Engine | None = None) -> None:
    """Load all four synthetic CSVs into postgres, in FK-safe order."""
    engine = engine or get_engine()

    print(f"Seeding from {DATA_DIR} ...")
    for table_name, date_cols in TABLES_IN_LOAD_ORDER:
        df = load_csv(table_name, date_cols)
        inserted = upsert_dataframe(engine, table_name, df)
        print(f"  {table_name:<16} {inserted:>6} new rows  (csv had {len(df)})")

    print("Done.")


if __name__ == "__main__":
    seed_all()
