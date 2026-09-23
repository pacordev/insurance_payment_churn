"""
Driver -> action mapping: turns a policy's top SHAP factor into a plausible
business intervention, answering the plan's fourth business question -
"what could the business do about it?"

This is static business logic (a lookup table), not row-level data, so per
churn_plan.md's Decisions it lives here as a plain Python dict rather than
getting its own DB table - churn_predict.py calls recommend_action() per
policy and writes the result into tbl_policy_risk_score.recommended_action.

Usage:
    python -m churn.recommend
"""

from __future__ import annotations

# tenure_days is deliberately left out - per churn_plan.md's milestone 8
# caveat, it's outcome-adjacent (churned policies' tenure is measured to
# their own cancel_date), not an independent behavior the business can act
# on, so recommend_action() skips it even when it's the top SHAP factor.
NON_ACTIONABLE_FEATURES = {"tenure_days"}

# only the behavior features from features.py get an entry here - one-hot
# encoded categoricals (insurance_product_*, payment_frequency_*) aren't in
# this dict either, so they fall through the same way tenure_days does
DRIVER_TO_ACTION: dict[str, str] = {
    "missed_invoice_ratio": (
        "Proactive outreach before the next invoice is due - this policy has "
        "a history of invoices going unpaid entirely."
    ),
    "failed_invoice_ratio": (
        "Prompt a payment-method update - repeated failed payment attempts "
        "usually mean an expired card or a bank-side decline."
    ),
    "late_invoice_ratio": (
        "Offer a flexible due date or grace-period plan - payments are "
        "consistently landing late rather than failing outright."
    ),
    "avg_days_late": (
        "Send payment reminders earlier in the billing cycle - payments "
        "are landing well past the arrears window on average."
    ),
    "max_days_late": (
        "Escalate to a retention call - at least one payment ran very "
        "late relative to the arrears window."
    ),
    "trend_failed_rate_delta": (
        "Escalate to the retention team - the failure rate has been "
        "getting worse in this policy's more recent invoices."
    ),
    "trend_late_rate_delta": (
        "Proactive check-in call - lateness has been trending worse in "
        "this policy's more recent invoices."
    ),
}

DEFAULT_ACTION = (
    "Monitor - no single actionable behavioral driver stands out, but the "
    "model still flags this policy as at risk."
)

# short, jargon-free versions of the same drivers - for the dashboard's
# business-facing "why is this policy flagged" bullets, where a raw
# feature name and a SHAP value ("failed_invoice_ratio: 0.067") would mean
# nothing to a non-technical reader
DRIVER_TO_DESCRIPTION: dict[str, str] = {
    "missed_invoice_ratio": "History of missed payments",
    "failed_invoice_ratio": "Repeated failed payment attempts",
    "late_invoice_ratio": "Payments consistently arriving late",
    "avg_days_late": "Payments landing well past the due window, on average",
    "max_days_late": "At least one payment ran very late",
    "trend_failed_rate_delta": "Payment failures have been getting worse recently",
    "trend_late_rate_delta": "Lateness has been trending worse recently",
}


def _actionable_risk_pushing_features(top_factors: list[dict]):
    """Shared filter behind recommend_action()/describe_factors(): drop tenure_days/categoricals and anything not pushing toward risk."""
    for factor in top_factors:
        feature = factor["feature"]
        value = factor["value"]
        if feature not in NON_ACTIONABLE_FEATURES and value > 0:
            yield feature


def recommend_action(top_factors: list[dict]) -> str:
    """
    Walk a policy's SHAP top_factors (ranked by magnitude, as
    churn_predict.py produces them) and return the action for the first
    one that's both actionable (not tenure_days or a one-hot categorical)
    and actually pushing the policy toward risk (positive shap value - a
    factor pulling the prediction the other way isn't something to
    intervene on). Falls back to a generic message if nothing qualifies.
    """
    for feature in _actionable_risk_pushing_features(top_factors):
        if feature in DRIVER_TO_ACTION:
            return DRIVER_TO_ACTION[feature]
    return DEFAULT_ACTION


def describe_factors(top_factors: list[dict], max_items: int = 3) -> list[str]:
    """
    Plain-language "why this policy is flagged" bullets - same filtering as
    recommend_action() (skip tenure_days/categoricals/negative values), but
    collects up to max_items descriptions instead of stopping at the first
    match, for the dashboard's business-facing policy view.
    """
    descriptions: list[str] = []
    for feature in _actionable_risk_pushing_features(top_factors):
        if feature in DRIVER_TO_DESCRIPTION:
            descriptions.append(DRIVER_TO_DESCRIPTION[feature])
        if len(descriptions) >= max_items:
            break
    return descriptions


if __name__ == "__main__":
    examples = [
        [{"feature": "tenure_days", "value": 0.31}, {"feature": "failed_invoice_ratio", "value": 0.07}],
        [{"feature": "trend_late_rate_delta", "value": 0.02}, {"feature": "insurance_product_Auto", "value": 0.01}],
        [{"feature": "tenure_days", "value": 0.4}, {"feature": "insurance_product_Home", "value": 0.05}],
    ]
    for factors in examples:
        print(f"top_factors={factors}\n  -> {recommend_action(factors)}\n")
