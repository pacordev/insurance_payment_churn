"""
Tests for explain.py. The ranking/formatting helpers get pure-array tests
(no model, no shap call needed). One integration-style test fits the actual
random forest + SHAP explainer against a small synthetic feature table, same
approach as test_churn_trainer.py/test_evaluation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.explain import (
    _clean_feature_names,
    aggregate_feature_importance,
    compute_shap_values,
    explain_policy,
    fit_random_forest,
)


def test_clean_feature_names_strips_the_columntransformer_prefix() -> None:
    raw_names = np.array(["numeric__failed_invoice_ratio", "categorical__insurance_product_Auto"])

    assert _clean_feature_names(raw_names) == ["failed_invoice_ratio", "insurance_product_Auto"]


def test_explain_policy_ranks_by_absolute_shap_value_not_signed_value() -> None:
    # one big negative contribution, several small positive ones - abs value should win
    X_transformed = pd.DataFrame(
        {"feature_a": [0.1], "feature_b": [0.2], "feature_c": [0.3]}
    )
    shap_values = np.array([[0.05, -0.9, 0.1]])

    top = explain_policy(shap_values, X_transformed, row_position=0, top_n=2)

    assert list(top["feature"]) == ["feature_b", "feature_c"]
    assert top.iloc[0]["shap_value"] == -0.9


def test_explain_policy_respects_top_n() -> None:
    X_transformed = pd.DataFrame({f"f{i}": [0.0] for i in range(10)})
    shap_values = np.array([np.arange(10, dtype=float)])

    top = explain_policy(shap_values, X_transformed, row_position=0, top_n=3)

    assert len(top) == 3
    # arange is already ascending, so the top 3 by magnitude are the last 3 features
    assert list(top["feature"]) == ["f9", "f8", "f7"]


def test_aggregate_feature_importance_ranks_by_mean_absolute_shap_value() -> None:
    X_transformed = pd.DataFrame({"noisy": [0.0, 0.0], "steady_driver": [0.0, 0.0]})
    # noisy: +5 then -5 (cancels out on average but not in magnitude);
    # steady_driver: a consistent +1 - mean |shap| should still rank noisy higher
    shap_values = np.array([[5.0, 1.0], [-5.0, 1.0]])

    importance = aggregate_feature_importance(shap_values, X_transformed)

    assert list(importance["feature"]) == ["noisy", "steady_driver"]
    assert importance.iloc[0]["mean_abs_shap_value"] == 5.0
    assert importance.iloc[1]["mean_abs_shap_value"] == 1.0


def _synthetic_feature_table_with_one_real_driver(n_per_class: int = 20, random_state: int = 0) -> pd.DataFrame:
    """
    Same insurance_product/payment_frequency and noisy everything-else for
    both groups - only failed_invoice_ratio actually differs by class - so
    there's exactly one feature a real driver-ranking should surface,
    unlike a fully collinear synthetic table where every column would tie.
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


def test_shap_pipeline_runs_end_to_end_and_flags_failed_invoice_ratio_as_a_top_driver() -> None:
    feature_table = _synthetic_feature_table_with_one_real_driver()

    pipeline, X_test, _y_test = fit_random_forest(feature_table, test_size=0.3)
    shap_values, X_transformed = compute_shap_values(pipeline, X_test)

    assert shap_values.shape == X_transformed.shape

    importance = aggregate_feature_importance(shap_values, X_transformed)
    # failed_invoice_ratio is the single biggest separator in the synthetic
    # data (0.0 vs 0.7) - the model should have picked up on it
    assert importance.iloc[0]["feature"] == "failed_invoice_ratio"
