"""
Tests for churn_trainer.py. All built from a small synthetic feature table
(no live db needed), same approach as test_features.py / test_segmentation.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.churn_trainer import CATEGORICAL_FEATURES, load_pipeline, save_pipeline, train_baseline_model
from churn.features import BEHAVIOR_FEATURE_COLUMNS


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


def test_train_baseline_model_returns_the_expected_metrics() -> None:
    feature_table = _synthetic_feature_table()

    _, metrics = train_baseline_model(feature_table, test_size=0.3)

    assert set(metrics) == {"precision", "recall", "roc_auc"}
    # cleanly separable synthetic data - a sane baseline should nail this
    assert metrics["roc_auc"] == 1.0


def test_pipeline_predicts_calibrated_looking_probabilities() -> None:
    feature_table = _synthetic_feature_table()

    pipeline, _ = train_baseline_model(feature_table, test_size=0.3)

    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    proba = pipeline.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # the risky group should score higher than the clean group on average
    assert proba[feature_table["is_churned"] == 1].mean() > proba[feature_table["is_churned"] == 0].mean()


def test_pipeline_handles_an_unseen_categorical_value_at_predict_time() -> None:
    # OneHotEncoder(handle_unknown="ignore") should keep this from blowing up
    # when scoring a policy with, say, a new insurance_product added later
    feature_table = _synthetic_feature_table()
    pipeline, _ = train_baseline_model(feature_table, test_size=0.3)

    unseen = feature_table.iloc[[0]].copy()
    unseen["insurance_product"] = "Umbrella"  # not in the synthetic training data
    X_unseen = unseen[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]

    proba = pipeline.predict_proba(X_unseen)[:, 1]
    assert len(proba) == 1


def test_save_and_load_pipeline_roundtrips_predictions(tmp_path: Path) -> None:
    feature_table = _synthetic_feature_table()
    pipeline, _ = train_baseline_model(feature_table, test_size=0.3)
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]

    saved_path = save_pipeline(pipeline, path=tmp_path / "model.joblib")
    loaded = load_pipeline(saved_path)

    assert saved_path.exists()
    np.testing.assert_array_equal(
        pipeline.predict_proba(X), loaded.predict_proba(X)
    )


def test_save_pipeline_creates_missing_parent_directories(tmp_path: Path) -> None:
    feature_table = _synthetic_feature_table()
    pipeline, _ = train_baseline_model(feature_table, test_size=0.3)

    nested_path = tmp_path / "nested" / "dir" / "model.joblib"
    saved_path = save_pipeline(pipeline, path=nested_path)

    assert saved_path == nested_path
    assert saved_path.exists()
