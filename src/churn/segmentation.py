"""
Unsupervised segmentation of policies by payment behavior.

Clusters policies purely on behavior features - never touches is_churned or
policy_status - then, as a sanity check, compares the clusters against the
hidden good/at_risk/churned profile that generate_data.py used to *generate*
the synthetic data (see syntethic_data/policy_profile_debug.csv). That file
is ground truth the model itself never sees, so a good match there is
evidence the clustering found real structure and not just noise.

Usage:
    python -m churn.segmentation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from churn.features import BEHAVIOR_FEATURE_COLUMNS, build_feature_table

# same folder seed_db.py reads the raw CSVs from
DEBUG_PROFILE_PATH = Path(__file__).resolve().parents[2] / "syntethic_data" / "policy_profile_debug.csv"

# alias kept for readability in this module - same columns churn_trainer.py trains on
SEGMENTATION_FEATURES = BEHAVIOR_FEATURE_COLUMNS


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


def label_clusters_by_risk(df_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw cluster ids (0/1/2, meaningless on their own) to human-readable
    risk labels, ranked by mean failed_invoice_ratio - purely descriptive,
    just naming clusters after the fact by how risky their behavior looks,
    not feeding anything back into how they were formed.
    """
    severity_rank = (
        df_with_cluster.groupby("cluster")["failed_invoice_ratio"].mean().sort_values().index
    )
    risk_names = ["Low risk", "Medium risk", "High risk"]
    # if n_clusters != 3 this still works, just falls back to generic names
    # past the first three ranks instead of guessing new adjectives
    names = risk_names + [f"Risk tier {i}" for i in range(3, len(severity_rank))]
    cluster_to_label = dict(zip(severity_rank, names))

    result = df_with_cluster.copy()
    result["segment_label"] = result["cluster"].map(cluster_to_label)
    return result


def load_hidden_profile(path: Path = DEBUG_PROFILE_PATH) -> pd.DataFrame:
    """Read the debug-only ground-truth profile generate_data.py wrote out."""
    return pd.read_csv(path)


def compare_to_hidden_profile(
    segmented: pd.DataFrame, hidden_profile: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """
    Cross-tab segment_label vs. the hidden good/at_risk/churned profile, plus
    the adjusted Rand index (0 = no better than random, 1 = perfect match) as
    a single number summarizing how well the two groupings line up.

    Note: generate_data.py overwrites policy_profile_debug.csv on every run,
    while seed_db.py accumulates policies across runs - so after more than
    one seed batch, hidden_profile usually only covers a subset of
    `segmented`. The inner merge below quietly keeps just the matching
    policies, which is fine for a spot-check but worth knowing about if the
    coverage looks low (see the printed match count in __main__).
    """
    merged = segmented.merge(hidden_profile, on="policy_num", how="inner")
    crosstab = pd.crosstab(merged["segment_label"], merged["hidden_profile"])
    ari = adjusted_rand_score(merged["hidden_profile"], merged["cluster"])
    return crosstab, ari


def build_segments(
    n_clusters: int = 3, random_state: int = 42, engine=None
) -> pd.DataFrame:
    """Full pipeline: feature table -> cluster -> risk-labeled segments."""
    feature_table = build_feature_table(engine)
    clustered = cluster_policies(feature_table, n_clusters=n_clusters, random_state=random_state)
    return label_clusters_by_risk(clustered)


if __name__ == "__main__":
    segments = build_segments()
    print(f"segmented {len(segments)} policies into {segments['segment_label'].nunique()} groups")
    print(segments["segment_label"].value_counts())

    hidden_profile = load_hidden_profile()
    matched = segments["policy_num"].isin(hidden_profile["policy_num"]).sum()
    print(f"\n{matched}/{len(segments)} policies have a hidden_profile match "
          f"(debug csv only covers the most recently generated batch)")

    crosstab, ari = compare_to_hidden_profile(segments, hidden_profile)
    print("\nsegment_label vs. hidden_profile (ground truth, unseen by clustering):")
    print(crosstab)
    print(f"\nadjusted Rand index: {ari:.3f}")
