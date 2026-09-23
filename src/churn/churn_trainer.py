"""
Baseline churn model: logistic regression on the per-policy feature table.

This is deliberately the simple version - a stratified split and a plain
logistic regression pipeline, just to establish baseline
precision/recall/ROC-AUC numbers. Calibration, threshold tuning and the
false-positive/false-negative cost discussion come later (evaluation.py);
trying random forest/GBM + SHAP explanations comes after that (explain.py).

Usage:
    python -m churn.churn_trainer
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features import BEHAVIOR_FEATURE_COLUMNS, build_feature_table

CATEGORICAL_FEATURES = ["insurance_product", "payment_frequency"]
LABEL_COLUMN = "is_churned"

# gitignored per the plan - just an artifact folder, not something to commit
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


def build_pipeline(random_state: int = 42, classifier: ClassifierMixin | None = None) -> Pipeline:
    """
    Preprocessing + classifier bundled as one pipeline, so the fitted object
    goes straight from raw feature-table columns to a prediction - no
    separate scaler/encoder to keep in sync at inference time later
    (churn_predict.py just calls .predict_proba on this). Defaults to the
    plain logistic regression baseline; pass a different classifier (e.g.
    build_random_forest_classifier()) to swap it out without duplicating the
    preprocessing step.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), BEHAVIOR_FEATURE_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    if classifier is None:
        # class_weight="balanced" since churn is the minority class (~24% of
        # policies) - a plain unweighted fit would lean too hard toward
        # always predicting "not churned" and still look decent on accuracy
        # alone, which is exactly the trap the plan calls out
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def build_random_forest_classifier(random_state: int = 42) -> RandomForestClassifier:
    """
    The "try a stronger model" half of milestone 8. Same class_weight
    reasoning as the logistic baseline; random forests also pair naturally
    with SHAP's TreeExplainer (exact and fast, unlike the general-purpose
    KernelExplainer a linear model would need).
    """
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)


def split_feature_table(
    feature_table: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split on the feature table - pulled out on its own
    so evaluation.py can fit a pipeline on the exact same split as here,
    instead of the two modules quietly drifting apart on split params.
    """
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    y = feature_table[LABEL_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_baseline_model(
    feature_table: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, dict[str, float]]:
    """Stratified train/test split, fit, and score a baseline logistic regression."""
    X_train, X_test, y_train, y_test = split_feature_table(
        feature_table, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return pipeline, metrics


def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    """
    Persist a fitted pipeline (preprocessing + classifier together, so
    there's nothing separate to keep in sync) via joblib, the persistence
    format the plan settled on - sklearn's pickle-based objects round-trip
    through it more reliably than plain pickle across versions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """
    Load a pipeline saved by save_pipeline() - what churn_predict.py will
    call at scoring time. joblib.load unpickles, but this only ever loads
    models/*.joblib that save_pipeline() itself wrote (gitignored, local-only
    artifact) - never anything from an untrusted or external source.
    """
    return joblib.load(path)


if __name__ == "__main__":
    feature_table = build_feature_table()
    _, metrics = train_baseline_model(feature_table)

    print(f"baseline logistic regression on {len(feature_table)} policies")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.3f}")
