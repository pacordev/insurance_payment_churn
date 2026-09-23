"""
Tests for segmentation.py. All built from plain DataFrames (no live db, no
KMeans randomness left to chance beyond a fixed random_state), same approach
as test_features.py.
"""

from __future__ import annotations

import pandas as pd

from churn.segmentation import (
    cluster_policies,
    compare_to_hidden_profile,
    label_clusters_by_risk,
)


def _feature_row(policy_num: str, **overrides: float) -> dict:
    # sensible "clean" defaults - individual tests override what matters to them
    row = {
        "policy_num": policy_num,
        "missed_invoice_ratio": 0.0,
        "failed_invoice_ratio": 0.0,
        "late_invoice_ratio": 0.0,
        "avg_days_late": 0.0,
        "max_days_late": 0.0,
        "trend_failed_rate_delta": 0.0,
        "trend_late_rate_delta": 0.0,
        "tenure_days": 365,
    }
    row.update(overrides)
    return row


def test_cluster_policies_separates_clean_from_risky_behavior() -> None:
    # 4 clean policies, 4 policies with heavy failed/missed payments and
    # short tenure - two obviously distinct behavior groups
    rows = [_feature_row(f"POL-CLEAN-{i}") for i in range(4)] + [
        _feature_row(
            f"POL-RISKY-{i}",
            missed_invoice_ratio=0.5,
            failed_invoice_ratio=0.6,
            late_invoice_ratio=0.3,
            avg_days_late=20.0,
            max_days_late=40.0,
            tenure_days=90,
        )
        for i in range(4)
    ]
    feature_table = pd.DataFrame(rows)

    clustered = cluster_policies(feature_table, n_clusters=2)

    clean_clusters = set(clustered[clustered["policy_num"].str.contains("CLEAN")]["cluster"])
    risky_clusters = set(clustered[clustered["policy_num"].str.contains("RISKY")]["cluster"])
    assert len(clean_clusters) == 1
    assert len(risky_clusters) == 1
    assert clean_clusters != risky_clusters


def test_label_clusters_by_risk_orders_by_failed_invoice_ratio() -> None:
    df = pd.DataFrame(
        [
            {"cluster": 0, "failed_invoice_ratio": 0.5},
            {"cluster": 0, "failed_invoice_ratio": 0.4},
            {"cluster": 1, "failed_invoice_ratio": 0.0},
            {"cluster": 1, "failed_invoice_ratio": 0.05},
            {"cluster": 2, "failed_invoice_ratio": 0.2},
        ]
    )

    labeled = label_clusters_by_risk(df)

    assert set(labeled.loc[labeled["cluster"] == 1, "segment_label"]) == {"Low risk"}
    assert set(labeled.loc[labeled["cluster"] == 2, "segment_label"]) == {"Medium risk"}
    assert set(labeled.loc[labeled["cluster"] == 0, "segment_label"]) == {"High risk"}


def test_compare_to_hidden_profile_perfect_match_gives_ari_one() -> None:
    segmented = pd.DataFrame(
        [
            {"policy_num": "POL-1", "cluster": 0, "segment_label": "Low risk"},
            {"policy_num": "POL-2", "cluster": 0, "segment_label": "Low risk"},
            {"policy_num": "POL-3", "cluster": 1, "segment_label": "High risk"},
            {"policy_num": "POL-4", "cluster": 1, "segment_label": "High risk"},
        ]
    )
    hidden_profile = pd.DataFrame(
        [
            {"policy_num": "POL-1", "hidden_profile": "good"},
            {"policy_num": "POL-2", "hidden_profile": "good"},
            {"policy_num": "POL-3", "hidden_profile": "churned"},
            {"policy_num": "POL-4", "hidden_profile": "churned"},
        ]
    )

    crosstab, ari = compare_to_hidden_profile(segmented, hidden_profile)

    assert ari == 1.0
    assert crosstab.loc["Low risk", "good"] == 2
    assert crosstab.loc["High risk", "churned"] == 2


def test_compare_to_hidden_profile_random_split_gives_low_ari() -> None:
    # clusters split 1/3-2/4 with no relationship to the hidden profile at all
    segmented = pd.DataFrame(
        [
            {"policy_num": "POL-1", "cluster": 0, "segment_label": "Low risk"},
            {"policy_num": "POL-2", "cluster": 1, "segment_label": "High risk"},
            {"policy_num": "POL-3", "cluster": 0, "segment_label": "Low risk"},
            {"policy_num": "POL-4", "cluster": 1, "segment_label": "High risk"},
        ]
    )
    hidden_profile = pd.DataFrame(
        [
            {"policy_num": "POL-1", "hidden_profile": "good"},
            {"policy_num": "POL-2", "hidden_profile": "good"},
            {"policy_num": "POL-3", "hidden_profile": "churned"},
            {"policy_num": "POL-4", "hidden_profile": "churned"},
        ]
    )

    _, ari = compare_to_hidden_profile(segmented, hidden_profile)

    assert ari < 1.0
