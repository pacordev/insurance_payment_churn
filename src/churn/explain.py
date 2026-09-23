"""
SHAP explainability for the churn model.

Milestone 8 pairs two things: trying a stronger model than the plain
logistic baseline, and explaining it. A random forest
(churn_trainer.build_random_forest_classifier) is the model this module
explains, for two reasons - it's usually a bit more predictive than plain
logistic regression, and SHAP's TreeExplainer is exact and fast for tree
models (vs. the general-purpose KernelExplainer a linear model would need,
which is slow and only approximate).

This gives the two views the plan calls out under "why" and "what
behaviors": explain_policy() decomposes one policy's prediction into its
top contributing factors ("why is *this* policy at risk?"), and
aggregate_feature_importance() ranks features by mean absolute SHAP value
across a whole batch ("what behaviors drive risk overall?").

Usage:
    python -m churn.explain
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churn.churn_trainer import build_pipeline, build_random_forest_classifier, split_feature_table

# is_churned == 1, matches LABEL_COLUMN / predict_proba[:, 1] everywhere else
CHURN_CLASS_INDEX = 1


def fit_random_forest(
    feature_table: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit the random forest pipeline on the usual split, return it plus the held-out set to explain."""
    X_train, X_test, y_train, y_test = split_feature_table(
        feature_table, test_size=test_size, random_state=random_state
    )
    pipeline = build_pipeline(random_state=random_state, classifier=build_random_forest_classifier(random_state))
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def _clean_feature_names(raw_names: np.ndarray) -> list[str]:
    """
    ColumnTransformer prefixes every output column with its transformer's
    name ("numeric__failed_invoice_ratio") - useful internally, noisy in a
    report meant for a business audience, so strip it back off.
    """
    return [name.split("__", 1)[-1] for name in raw_names]


def transform_for_explaining(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through the fitted preprocessing step - SHAP explains this space (readable columns), not the raw dataframe."""
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(X)
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    SHAP values for the churn (positive) class: one row per policy in X, one
    column per (cleaned) transformed feature, same shape as X_transformed -
    shap_values[i, j] is exactly feature j's contribution to policy i's
    churn score.
    """
    X_transformed = transform_for_explaining(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    raw_values = explainer.shap_values(X_transformed)
    # shap's binary-classifier output shape has changed across versions:
    # older releases return a list of two 2D arrays (one per class), newer
    # ones a single (n_samples, n_features, n_classes) array - normalize
    # both down to just the churn-class 2D array
    if isinstance(raw_values, list):
        churn_shap_values = raw_values[CHURN_CLASS_INDEX]
    else:
        churn_shap_values = raw_values[:, :, CHURN_CLASS_INDEX]
    return churn_shap_values, X_transformed


def explain_policy(
    shap_values: np.ndarray, X_transformed: pd.DataFrame, row_position: int, top_n: int = 5
) -> pd.DataFrame:
    """
    Top contributing factors for one policy, addressed by its row position
    in X_transformed (not policy_num - look up the position first via the
    original feature table's index, see the __main__ block below).
    Positive shap_value pushed the prediction toward "at risk", negative
    pushed it toward "safe".
    """
    contributions = pd.DataFrame(
        {
            "feature": X_transformed.columns,
            "value": X_transformed.iloc[row_position].to_numpy(),
            "shap_value": shap_values[row_position],
        }
    )
    ranked = contributions.reindex(contributions["shap_value"].abs().sort_values(ascending=False).index)
    return ranked.head(top_n).reset_index(drop=True)


def aggregate_feature_importance(shap_values: np.ndarray, X_transformed: pd.DataFrame) -> pd.DataFrame:
    """
    "What behaviors drive risk overall?" - mean absolute SHAP value per
    feature across every policy passed in, ranked descending. The aggregate
    counterpart to explain_policy()'s per-policy view.
    """
    importance = pd.DataFrame(
        {
            "feature": X_transformed.columns,
            "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from churn.features import build_feature_table

    feature_table = build_feature_table()
    pipeline, X_test, y_test = fit_random_forest(feature_table)
    shap_values, X_transformed = compute_shap_values(pipeline, X_test)

    print(f"SHAP explanations for {len(X_test)} held-out policies\n")

    print("-- aggregate feature importance (mean |SHAP value|, all held-out policies) --")
    print(aggregate_feature_importance(shap_values, X_transformed).to_string(index=False))

    # a concrete example: the held-out policy the model thinks is riskiest
    churn_proba = pipeline.predict_proba(X_test)[:, CHURN_CLASS_INDEX]
    riskiest_position = int(np.argmax(churn_proba))
    riskiest_policy_num = feature_table.loc[X_test.index[riskiest_position], "policy_num"]

    print(f"\n-- top factors for the riskiest held-out policy ({riskiest_policy_num}, "
          f"predicted risk {churn_proba[riskiest_position]:.1%}) --")
    print(explain_policy(shap_values, X_transformed, riskiest_position).to_string(index=False))
