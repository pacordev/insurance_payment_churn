"""
Tests for evaluation.py. Most of these work off hand-built y_true/y_proba
arrays (no model, no db needed) so the metric math and the cost-weighted
threshold logic are pinned down exactly. One integration-style test at the
bottom runs full_evaluation() against a small synthetic feature table, same
approach as test_churn_trainer.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.evaluation import (
    calibration_table,
    compute_metrics,
    confusion_counts,
    full_evaluation,
    pick_cost_weighted_threshold,
    save_evaluation_report,
    threshold_sweep,
)


def test_compute_metrics_on_a_perfect_classifier() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_proba)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_confusion_counts_matches_a_hand_counted_example() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.6, 0.2, 0.3, 0.9])  # one false positive, one false negative at 0.5

    counts = confusion_counts(y_true, y_proba, threshold=0.5)

    assert counts == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }


def test_higher_threshold_trades_recall_for_precision() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1])
    y_proba = np.array([0.3, 0.55, 0.6, 0.7, 0.9])

    loose = compute_metrics(y_true, y_proba, threshold=0.1)
    strict = compute_metrics(y_true, y_proba, threshold=0.65)

    # a very low threshold flags everyone -> catches every real churner
    assert loose["recall"] == 1.0
    # a stricter one misses a borderline churner -> recall drops
    assert strict["recall"] < loose["recall"]


def test_calibration_table_has_the_expected_shape() -> None:
    rng = np.random.default_rng(0)
    y_proba = rng.uniform(0, 1, size=100)
    y_true = pd.Series((rng.uniform(0, 1, size=100) < y_proba).astype(int))

    table = calibration_table(y_true, y_proba, n_bins=5)

    assert set(table.columns) == {"mean_predicted_probability", "observed_churn_rate"}
    # quantile binning can merge sparse buckets, so this is an upper bound
    assert len(table) <= 5


def test_threshold_sweep_covers_exactly_the_requested_thresholds() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])

    sweep = threshold_sweep(y_true, y_proba, thresholds=[0.3, 0.5, 0.7])

    assert list(sweep["threshold"]) == [0.3, 0.5, 0.7]
    expected_columns = {"precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive", "false_negative"}
    assert expected_columns <= set(sweep.columns)


def test_cost_weighted_threshold_favors_recall_when_false_negatives_are_expensive() -> None:
    # a real churner sits right at 0.45 - only a threshold below that catches
    # them, at the cost of one extra false positive at 0.55
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.55, 0.45, 0.9])

    best_threshold, _ = pick_cost_weighted_threshold(
        y_true, y_proba, cost_fp=1.0, cost_fn=20.0, thresholds=[0.4, 0.5, 0.6]
    )

    # missing the churner costs 20, tolerating the false positive only costs 1
    assert best_threshold == 0.4


def test_cost_weighted_threshold_favors_precision_when_false_positives_are_expensive() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.55, 0.45, 0.9])

    best_threshold, _ = pick_cost_weighted_threshold(
        y_true, y_proba, cost_fp=20.0, cost_fn=1.0, thresholds=[0.4, 0.5, 0.6]
    )

    # now the false positive at 0.55 is the expensive mistake to avoid
    assert best_threshold == 0.6


def _synthetic_feature_table(n_per_class: int = 15) -> pd.DataFrame:
    """Two cleanly separable groups of policies - clean/good vs. risky/churned."""
    rows: list[dict] = []
    for i in range(n_per_class):
        rows.append({
            "policy_num": f"POL-GOOD-{i}",
            "insurance_product": "Auto",
            "payment_frequency": "Monthly",
            "missed_invoice_ratio": 0.0,
            "failed_invoice_ratio": 0.0,
            "late_invoice_ratio": 0.0,
            "avg_days_late": 0.0,
            "max_days_late": 0.0,
            "trend_failed_rate_delta": 0.0,
            "trend_late_rate_delta": 0.0,
            "tenure_days": 365,
            "is_churned": 0,
        })
        rows.append({
            "policy_num": f"POL-CHURN-{i}",
            "insurance_product": "Home",
            "payment_frequency": "Quarterly",
            "missed_invoice_ratio": 0.6,
            "failed_invoice_ratio": 0.7,
            "late_invoice_ratio": 0.4,
            "avg_days_late": 25.0,
            "max_days_late": 50.0,
            "trend_failed_rate_delta": 0.3,
            "trend_late_rate_delta": 0.2,
            "tenure_days": 100,
            "is_churned": 1,
        })
    return pd.DataFrame(rows)


def test_full_evaluation_runs_end_to_end_on_a_synthetic_feature_table() -> None:
    feature_table = _synthetic_feature_table()

    result = full_evaluation(feature_table, test_size=0.3)

    # cleanly separable synthetic data - a sane baseline should nail this
    assert result["default_threshold_metrics"]["roc_auc"] == 1.0
    assert 0.0 <= result["cost_weighted_threshold"] <= 1.0
    assert set(result["cost_weighted_metrics"]) >= {"precision", "recall", "f1"}
    assert not result["threshold_sweep"].empty
    assert set(result["calibration"].columns) == {"mean_predicted_probability", "observed_churn_rate"}


def test_save_evaluation_report_writes_the_expected_files(tmp_path: Path) -> None:
    feature_table = _synthetic_feature_table()
    result = full_evaluation(feature_table, test_size=0.3)

    paths = save_evaluation_report(result, output_dir=tmp_path)

    assert paths["report"].exists()
    assert paths["threshold_sweep"].exists()
    assert paths["calibration"].exists()

    report = json.loads(paths["report"].read_text())
    assert report["default_threshold_metrics"]["roc_auc"] == 1.0
    assert "cost_weighted_threshold" in report
    assert "cost_weighted_metrics" in report
