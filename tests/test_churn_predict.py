"""
Tests for churn_predict.py. Fits a small random forest on a synthetic
feature table (no live db) and scores it end-to-end, same approach as
test_explain.py. write_risk_scores()/run_batch_scoring() talk to postgres
directly and aren't unit tested here, matching how seed_db.py's db-writing
functions are handled elsewhere in this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.churn_predict import (
    compute_batch_feature_importance,
    filter_active_policies,
    save_feature_importance,
    score_all_policies,
)
from churn.churn_trainer import CATEGORICAL_FEATURES, LABEL_COLUMN, build_pipeline, build_random_forest_classifier
from churn.features import BEHAVIOR_FEATURE_COLUMNS


def _synthetic_feature_table_with_one_real_driver(n_per_class: int = 20, random_state: int = 0) -> pd.DataFrame:
    """
    Same insurance_product/payment_frequency and noisy everything-else for
    both groups - only failed_invoice_ratio actually differs by class - so
    aggregate/per-policy driver rankings have exactly one real signal to
    find, same reasoning as test_explain.py's synthetic table.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []
    for i in range(n_per_class):
        for policy_prefix, is_churned, failed_ratio_range in (
            ("POL-GOOD", 0, (0.0, 0.1)),
            ("POL-CHURN", 1, (0.6, 0.9)),
        ):
            rows.append({
                "policy_num": f"{policy_prefix}-{i}",
                "insurance_product": "Auto",
                "payment_frequency": "Monthly",
                "missed_invoice_ratio": rng.uniform(0.0, 0.3),
                "failed_invoice_ratio": rng.uniform(*failed_ratio_range),
                "late_invoice_ratio": rng.uniform(0.0, 0.3),
                "avg_days_late": rng.uniform(0.0, 10.0),
                "max_days_late": rng.uniform(0.0, 15.0),
                "trend_failed_rate_delta": rng.uniform(-0.1, 0.1),
                "trend_late_rate_delta": rng.uniform(-0.1, 0.1),
                "tenure_days": int(rng.integers(200, 400)),
                "is_churned": is_churned,
            })
    return pd.DataFrame(rows)


def _fitted_pipeline(feature_table: pd.DataFrame):
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    y = feature_table[LABEL_COLUMN]
    pipeline = build_pipeline(classifier=build_random_forest_classifier())
    pipeline.fit(X, y)
    return pipeline


def test_filter_active_policies_keeps_only_active_rows() -> None:
    feature_table = pd.DataFrame({
        "policy_num": ["POL-1", "POL-2", "POL-3"],
        "policy_status": ["Active", "Cancelled", "Expired"],
        "tenure_days": [100, 200, 300],
    })

    filtered = filter_active_policies(feature_table)

    assert list(filtered["policy_num"]) == ["POL-1"]


def test_filter_active_policies_returns_empty_frame_when_none_are_active() -> None:
    feature_table = pd.DataFrame({
        "policy_num": ["POL-1", "POL-2"],
        "policy_status": ["Cancelled", "Expired"],
    })

    filtered = filter_active_policies(feature_table)

    assert filtered.empty


def test_score_all_policies_returns_one_row_per_policy_with_expected_columns() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table)

    assert len(scores) == len(feature_table)
    assert set(scores.columns) == {
        "policy_num", "scored_at", "churn_probability", "segment_label", "top_factors", "recommended_action",
    }
    assert set(scores["policy_num"]) == set(feature_table["policy_num"])


def test_score_all_policies_probabilities_are_in_range_and_separate_the_groups() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table)

    assert ((scores["churn_probability"] >= 0) & (scores["churn_probability"] <= 1)).all()
    merged = scores.merge(feature_table[["policy_num", "is_churned"]], on="policy_num")
    churn_avg = merged.loc[merged["is_churned"] == 1, "churn_probability"].mean()
    good_avg = merged.loc[merged["is_churned"] == 0, "churn_probability"].mean()
    assert churn_avg > good_avg


def test_score_all_policies_shares_one_scored_at_across_the_batch() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table)

    assert scores["scored_at"].nunique() == 1


def test_score_all_policies_top_factors_has_the_expected_shape() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table, top_n_factors=2)

    first_row_factors = scores["top_factors"].iloc[0]
    assert len(first_row_factors) == 2
    assert set(first_row_factors[0]) == {"feature", "value"}
    assert isinstance(first_row_factors[0]["feature"], str)
    assert isinstance(first_row_factors[0]["value"], float)


def test_score_all_policies_top_factor_across_the_batch_is_dominated_by_the_real_driver() -> None:
    # per-row top factor can occasionally vary due to tree interactions, but
    # across the whole risky group failed_invoice_ratio (the one real
    # signal) should be the most common #1 factor by a wide margin
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table, top_n_factors=1)

    churn_scores = scores[scores["policy_num"].str.startswith("POL-CHURN")]
    top_feature_counts = churn_scores["top_factors"].apply(lambda factors: factors[0]["feature"]).value_counts()
    assert top_feature_counts.idxmax() == "failed_invoice_ratio"


def test_score_all_policies_fills_in_recommended_action_from_the_top_factor() -> None:
    from churn.recommend import DRIVER_TO_ACTION

    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    scores = score_all_policies(pipeline, feature_table)

    # the most commonly recommended action for the risky group should be
    # the one tied to failed_invoice_ratio, the one real driver
    churn_scores = scores[scores["policy_num"].str.startswith("POL-CHURN")]
    assert churn_scores["recommended_action"].mode().iloc[0] == DRIVER_TO_ACTION["failed_invoice_ratio"]


def test_compute_batch_feature_importance_ranks_the_real_driver_first() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)

    importance = compute_batch_feature_importance(pipeline, feature_table)

    assert importance.iloc[0]["feature"] == "failed_invoice_ratio"


def test_save_feature_importance_writes_a_csv(tmp_path: Path) -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()
    pipeline = _fitted_pipeline(feature_table)
    importance = compute_batch_feature_importance(pipeline, feature_table)

    path = save_feature_importance(importance, path=tmp_path / "feature_importance.csv")

    assert path.exists()
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["feature", "mean_abs_shap_value"]
