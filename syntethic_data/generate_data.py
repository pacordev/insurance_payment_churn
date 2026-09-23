#!/usr/bin/env python3
"""
Churn prediction synthetic data generator.

Generates referentially consistent data across:
    tbl_customer → tbl_policies → tbl_invoice → tbl_payhistory

Customer behaviour profiles
───────────────────────────
  good     – pays on time; policy stays active or expires naturally
  at_risk  – occasional late / failed payments; still active
  churned  – increasing failures leading to policy cancellation

Payment outcomes deteriorate over time for churned customers so the
temporal signal is present for ML feature engineering.

Also writes policy_profile_debug.csv (policy_num -> hidden_profile).
That's ground truth for validating unsupervised segmentation later - it's
NOT one of the tbl_*.csv files, so seed_db.py never loads it into
Postgres or lets it leak into the feature table.

Usage:
    python generate_data.py
    python generate_data.py --customers 120 --seed 7 --out-dir ./data

Dependencies:
    pip install faker
"""

import argparse
import csv
import random
import uuid
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TODAY     = date.today()
MIN_START = date(2022, 1, 1)
MAX_START = TODAY - timedelta(days=183)          # at least 6 months of history

INSURANCE_PRODUCTS   = ['Auto', 'Home', 'Life', 'Health', 'Commercial Property', 'Renters', 'Umbrella']
PAYMENT_FREQUENCIES  = ['Monthly', 'Quarterly', 'Semi-Annual']
FREQ_MONTHS          = {'Monthly': 1, 'Quarterly': 3, 'Semi-Annual': 6}
INVOICE_TYPES        = ['Premium', 'Adjustment', 'Refund']
INVOICE_TYPE_WEIGHTS = [85, 12, 3]

# Profile mix (weights must sum to a consistent ratio; they are normalised internally)
PROFILES        = ['good', 'at_risk', 'churned']
PROFILE_WEIGHTS = [40, 35, 25]

# Payment outcome probabilities per profile
# 'missed' means no transaction is recorded for that invoice
PAYMENT_OUTCOMES: dict[str, dict[str, float]] = {
    'good':    {'on_time': 0.90, 'late': 0.07, 'failed': 0.02, 'missed': 0.01},
    'at_risk': {'on_time': 0.55, 'late': 0.22, 'failed': 0.15, 'missed': 0.08},
    'churned': {'on_time': 0.25, 'late': 0.18, 'failed': 0.32, 'missed': 0.25},
}

# Probability that a failed payment is retried successfully, per profile.
# For churned customers this is further reduced as progress → 1.
RETRY_SUCCESS_RATE: dict[str, float] = {
    'good':    0.80,   # almost always recovers
    'at_risk': 0.50,   # recovers half the time
    'churned': 0.20,   # rarely recovers, and gets worse over time
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def random_date(start: date, end: date) -> date:
    delta = max((end - start).days, 0)
    return start + timedelta(days=random.randint(0, delta))

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_customers(fake: Faker, n: int) -> tuple[list[dict], dict[str, str]]:
    """Return (rows, customer_id → profile) with profile pre-assigned."""
    assigned = random.choices(PROFILES, weights=PROFILE_WEIGHTS, k=n)
    rows: list[dict] = []
    cust_profile: dict[str, str] = {}

    for profile in assigned:
        cid = gen_id('CUST')
        rows.append({
            'customer_id':  cid,
            'tax_id':       fake.numerify('###-##-####'),
            'c_first_name': fake.first_name(),
            'c_last_name':  fake.last_name(),
        })
        cust_profile[cid] = profile

    return rows, cust_profile


def generate_policies(
    customers: list[dict],
    cust_profile: dict[str, str],
) -> tuple[list[dict], dict[str, str]]:
    """Return (rows, policy_num → profile)."""
    rows: list[dict] = []
    policy_profile: dict[str, str] = {}

    for cust in customers:
        cid     = cust['customer_id']
        profile = cust_profile[cid]

        for _ in range(random.randint(1, 2)):
            start = random_date(MIN_START, MAX_START)
            end   = start + timedelta(days=365)

            if profile == 'churned':
                # Cancel between 90 days in and the lesser of (end, today)
                earliest_cancel = start + timedelta(days=90)
                latest_cancel   = min(end, TODAY)
                if earliest_cancel > latest_cancel:
                    earliest_cancel = start + timedelta(days=30)
                cancel_date = random_date(earliest_cancel, latest_cancel)
                status      = 'Cancelled'
            else:
                cancel_date = None
                status      = 'Active' if end >= TODAY else 'Expired'

            policy_num = gen_id('POL')
            rows.append({
                'insurance_product':    random.choice(INSURANCE_PRODUCTS),
                'customer_id':          cid,
                'policy_num':           policy_num,
                'effective_start_date': start,
                'effective_end_date':   end,
                'payment_frequency':    random.choice(PAYMENT_FREQUENCIES),
                'policy_status':        status,
                'cancel_date':          cancel_date,
            })
            policy_profile[policy_num] = profile

    return rows, policy_profile


def generate_invoices(policies: list[dict]) -> list[dict]:
    """One invoice per frequency period up to the relevant cutoff date."""
    rows: list[dict] = []

    for pol in policies:
        months    = FREQ_MONTHS[pol['payment_frequency']]
        pol_start = pol['effective_start_date']
        pol_end   = pol['effective_end_date']
        status    = pol['policy_status']
        cancel    = pol['cancel_date']

        # Generate invoices only up to the natural boundary for this policy
        if status == 'Cancelled' and cancel:
            cutoff = cancel
        elif status == 'Active':
            cutoff = TODAY
        else:                       # Expired
            cutoff = pol_end

        base_premium = round(random.uniform(80.0, 1800.0), 2)
        installment  = 1
        inv_start    = pol_start

        while inv_start < cutoff:
            inv_end = min(inv_start + timedelta(days=30 * months), pol_end)
            rows.append({
                'policy_num':           pol['policy_num'],
                'invoice_num':          gen_id('INV'),
                'invoice_type':         random.choices(INVOICE_TYPES, weights=INVOICE_TYPE_WEIGHTS)[0],
                'invoice_start_date':   inv_start,
                'invoice_end_date':     inv_end,
                'days_arrears_allowed': random.choice([30, 60, 90]),
                'installment_num':      installment,
                'premium':              base_premium,
            })
            inv_start   = inv_end
            installment += 1

    return rows


def _pick_outcome(profile: str, progress: float) -> str:
    """
    Sample a payment outcome for one invoice.

    For churned customers, 'missed' and 'failed' increase as the policy
    approaches cancellation (progress goes 0 → 1), mirroring real churn
    behaviour so temporal features carry a signal.
    """
    weights = dict(PAYMENT_OUTCOMES[profile])

    if profile == 'churned':
        weights['missed']  = min(0.65, weights['missed']  + progress * 0.40)
        weights['failed']  = min(0.45, weights['failed']  + progress * 0.13)
        weights['on_time'] = max(0.05, weights['on_time'] - progress * 0.20)
        weights['late']    = max(0.05, weights['late']    - progress * 0.10)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    return random.choices(list(weights), weights=list(weights.values()))[0]


def generate_payhistory(
    invoices: list[dict],
    policy_profile: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []

    # Group by policy so we can track position within the policy lifespan
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for inv in invoices:
        by_policy[inv['policy_num']].append(inv)

    for policy_num, pol_invoices in by_policy.items():
        profile = policy_profile.get(policy_num, 'good')
        n       = len(pol_invoices)

        for idx, inv in enumerate(pol_invoices):
            progress = idx / max(n - 1, 1)      # 0.0 (first) → 1.0 (last)
            outcome  = _pick_outcome(profile, progress)

            if outcome == 'missed':
                continue                          # no record for this invoice

            inv_start    = inv['invoice_start_date']
            days_arrears = inv['days_arrears_allowed']

            if outcome == 'on_time':
                rows.append({
                    'transaction_id':         gen_id('TXN'),
                    'invoice_num':            inv['invoice_num'],
                    'premium':                inv['premium'],
                    'pay_transaction_date':   inv_start + timedelta(days=random.randint(1, 10)),
                    'pay_transaction_status': 'Paid',
                })

            elif outcome == 'late':
                late_days = random.randint(days_arrears + 1, days_arrears + 30)
                rows.append({
                    'transaction_id':         gen_id('TXN'),
                    'invoice_num':            inv['invoice_num'],
                    'premium':                inv['premium'],
                    'pay_transaction_date':   inv_start + timedelta(days=late_days),
                    'pay_transaction_status': 'Paid Late',
                })

            elif outcome == 'failed':
                fail_date = inv_start + timedelta(days=random.randint(1, 15))
                rows.append({
                    'transaction_id':         gen_id('TXN'),
                    'invoice_num':            inv['invoice_num'],
                    'premium':                inv['premium'],
                    'pay_transaction_date':   fail_date,
                    'pay_transaction_status': 'Failed',
                })
                # Retry rate is profile-dependent; for churned it also drops
                # as the policy nears cancellation (progress → 1).
                base_retry = RETRY_SUCCESS_RATE[profile]
                retry_prob = base_retry * (1 - progress * 0.75) if profile == 'churned' else base_retry
                if random.random() < retry_prob:
                    rows.append({
                        'transaction_id':         gen_id('TXN'),
                        'invoice_num':            inv['invoice_num'],
                        'premium':                inv['premium'],
                        'pay_transaction_date':   fail_date + timedelta(days=random.randint(3, 10)),
                        'pay_transaction_status': 'Paid',
                    })

    return rows

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name:<30} {len(rows):>8,} rows")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate churn prediction synthetic data.')
    parser.add_argument('--customers', type=int, default=120,
                        help='Number of customers (default: 120)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--out-dir',   type=str, default='.',
                        help='Output directory for CSVs (default: current dir)')
    args = parser.parse_args()

    fake = Faker()
    Faker.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating churn prediction data  (seed={args.seed}, customers={args.customers}) ...\n")

    customers,  cust_profile   = generate_customers(fake, args.customers)
    policies,   policy_profile = generate_policies(customers, cust_profile)
    invoices                   = generate_invoices(policies)
    payhistory                 = generate_payhistory(invoices, policy_profile)

    profile_counts = Counter(cust_profile.values())
    print("Customer profiles:")
    for p in PROFILES:
        print(f"  {p:<12} {profile_counts[p]:>4} customers")

    print("\nRecord counts:")
    print(f"  tbl_customer:   {len(customers):>8,}")
    print(f"  tbl_policies:   {len(policies):>8,}")
    print(f"  tbl_invoice:    {len(invoices):>8,}")
    print(f"  tbl_payhistory: {len(payhistory):>8,}")

    print(f"\nWriting CSVs to '{out_dir}' ...")
    write_csv(customers,  out_dir / 'tbl_customer.csv')
    write_csv(policies,   out_dir / 'tbl_policies.csv')
    write_csv(invoices,   out_dir / 'tbl_invoice.csv')
    write_csv(payhistory, out_dir / 'tbl_payhistory.csv')

    # Debug-only: the hidden profile that drove each policy's payment
    # behaviour. Deliberately NOT one of the tbl_*.csv files above, so
    # seed_db.py (which only loads that fixed list) never lets it into
    # Postgres or the feature table - it's ground truth for checking
    # unsupervised segmentation later, not a feature the model gets to see.
    profile_debug_rows = [
        {'policy_num': policy_num, 'hidden_profile': profile}
        for policy_num, profile in policy_profile.items()
    ]
    write_csv(profile_debug_rows, out_dir / 'policy_profile_debug.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()
