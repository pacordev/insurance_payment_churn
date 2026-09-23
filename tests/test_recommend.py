"""
Tests for recommend.py. Pure lookup logic, all built from hand-built
top_factors lists (the exact shape churn_predict.py produces) - no model,
no db needed.
"""

from __future__ import annotations

from churn.recommend import DEFAULT_ACTION, DRIVER_TO_ACTION, DRIVER_TO_DESCRIPTION, describe_factors, recommend_action


def test_recommend_action_maps_the_top_factor_when_actionable() -> None:
    top_factors = [{"feature": "failed_invoice_ratio", "value": 0.4}]

    action = recommend_action(top_factors)

    assert action == DRIVER_TO_ACTION["failed_invoice_ratio"]


def test_recommend_action_skips_tenure_days_and_falls_through_to_the_next_factor() -> None:
    # tenure_days is the top SHAP driver in the live data, but it's
    # outcome-adjacent, not an actionable behavior - see churn_plan.md
    top_factors = [
        {"feature": "tenure_days", "value": 0.31},
        {"feature": "missed_invoice_ratio", "value": 0.1},
    ]

    action = recommend_action(top_factors)

    assert action == DRIVER_TO_ACTION["missed_invoice_ratio"]


def test_recommend_action_skips_a_factor_pushing_toward_safe_not_risk() -> None:
    # negative shap value = pulled the prediction toward "not churned" -
    # not something to intervene on, even if it's an otherwise known driver
    top_factors = [
        {"feature": "failed_invoice_ratio", "value": -0.2},
        {"feature": "late_invoice_ratio", "value": 0.05},
    ]

    action = recommend_action(top_factors)

    assert action == DRIVER_TO_ACTION["late_invoice_ratio"]


def test_recommend_action_skips_unknown_one_hot_categorical_features() -> None:
    top_factors = [
        {"feature": "insurance_product_Auto", "value": 0.2},
        {"feature": "payment_frequency_Quarterly", "value": 0.15},
        {"feature": "trend_failed_rate_delta", "value": 0.03},
    ]

    action = recommend_action(top_factors)

    assert action == DRIVER_TO_ACTION["trend_failed_rate_delta"]


def test_recommend_action_falls_back_to_default_when_nothing_qualifies() -> None:
    top_factors = [
        {"feature": "tenure_days", "value": 0.31},
        {"feature": "insurance_product_Auto", "value": 0.1},
    ]

    action = recommend_action(top_factors)

    assert action == DEFAULT_ACTION


def test_recommend_action_handles_an_empty_top_factors_list() -> None:
    assert recommend_action([]) == DEFAULT_ACTION


def test_describe_factors_returns_plain_language_bullets_in_ranked_order() -> None:
    top_factors = [
        {"feature": "failed_invoice_ratio", "value": 0.4},
        {"feature": "missed_invoice_ratio", "value": 0.2},
    ]

    descriptions = describe_factors(top_factors)

    assert descriptions == [
        DRIVER_TO_DESCRIPTION["failed_invoice_ratio"],
        DRIVER_TO_DESCRIPTION["missed_invoice_ratio"],
    ]


def test_describe_factors_skips_tenure_days_categoricals_and_negative_values() -> None:
    top_factors = [
        {"feature": "tenure_days", "value": 0.31},
        {"feature": "insurance_product_Auto", "value": 0.2},
        {"feature": "failed_invoice_ratio", "value": -0.1},
        {"feature": "late_invoice_ratio", "value": 0.05},
    ]

    descriptions = describe_factors(top_factors)

    assert descriptions == [DRIVER_TO_DESCRIPTION["late_invoice_ratio"]]


def test_describe_factors_respects_max_items() -> None:
    top_factors = [
        {"feature": "failed_invoice_ratio", "value": 0.4},
        {"feature": "missed_invoice_ratio", "value": 0.3},
        {"feature": "late_invoice_ratio", "value": 0.2},
    ]

    descriptions = describe_factors(top_factors, max_items=2)

    assert len(descriptions) == 2


def test_describe_factors_handles_an_empty_top_factors_list() -> None:
    assert describe_factors([]) == []
