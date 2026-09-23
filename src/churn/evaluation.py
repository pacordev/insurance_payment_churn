"""
Model evaluation as a decision problem, not just a training exercise.

Accuracy alone is a trap here - churn is ~24% of policies, so a model that
just predicts "not churned" for everyone already looks decent on accuracy
and is completely useless. This module goes further: precision/recall/F1/
ROC-AUC/PR-AUC, a confusion matrix, a calibration table, and a threshold
sweep - then uses that sweep to pick an operating threshold based on the
actual business cost of the two kinds of mistakes, instead of defaulting
to 0.5:
  - false positive: a policy flagged as at-risk that wouldn't have churned
    - cost is a wasted retention outreach (an email/call/discount offer).
  - false negative: a policy that actually churns but the model misses -
    cost is a lost customer with zero warning, presumably worth a lot more
    than one wasted outreach.
Since cost_fn > cost_fp in most retention scenarios, the cost-weighted
threshold usually ends up lower than 0.5 - the model should flag more
people than a naive threshold would, because missing a real churner is
so much more expensive than one extra outreach.

Milestone 9 adds persistence on top: save_evaluation_report() writes
full_evaluation()'s numbers to disk (JSON for the scalar metrics, CSV for
the two tables) next to the model file churn_trainer.save_pipeline() saves,
so churn_predict.py and the dashboard can show how the saved model performs
without retraining it.

Usage:
    python -m churn.evaluation
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.churn_trainer import MODEL_DIR, build_pipeline, split_feature_table

DEFAULT_THRESHOLD = 0.5
# retention email/call vs. a lost customer with no warning - cost_fn should
# always be meaningfully bigger than cost_fp for this business, these are
# just placeholder units (not real dollars) to demonstrate the trade-off
DEFAULT_COST_FALSE_POSITIVE = 1.0
DEFAULT_COST_FALSE_NEGATIVE = 5.0


def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Precision/recall/F1 at a given threshold, plus threshold-independent ROC-AUC/PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def confusion_counts(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, int]:
    """Raw TP/FP/TN/FN counts at a given threshold - the numbers the cost trade-off runs on."""
    y_pred = (y_proba >= threshold).astype(int)
    # labels=[0, 1] pins the matrix shape even if one class is missing from y_pred
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def calibration_table(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions by predicted probability and compare against the
    observed churn rate in each bucket - a well-calibrated model has these
    roughly matching (a bucket predicted around 70% risk should actually
    churn about 70% of the time). Quantile bins so each bucket gets a
    similar number of policies rather than being skewed by wherever the
    scores happen to cluster.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted_value,
            "observed_churn_rate": fraction_of_positives,
        }
    )


def threshold_sweep(
    y_true: pd.Series, y_proba: np.ndarray, thresholds: Sequence[float] | None = None
) -> pd.DataFrame:
    """One row per threshold: metrics + confusion counts, the raw material for picking an operating point."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = [
        {**compute_metrics(y_true, y_proba, threshold=t), **confusion_counts(y_true, y_proba, threshold=t)}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def pick_cost_weighted_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
    thresholds: Sequence[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep thresholds, weight each one's mistakes by their actual business
    cost (expected_cost = false_positives * cost_fp + false_negatives *
    cost_fn), and return whichever threshold minimizes that - not whichever
    gets the best F1. This is the "decision problem" framing: 0.5 is just a
    default that assumes both error types cost the same, which usually isn't
    true here.
    """
    sweep = threshold_sweep(y_true, y_proba, thresholds)
    sweep["expected_cost"] = sweep["false_positive"] * cost_fp + sweep["false_negative"] * cost_fn
    best_row = sweep.loc[sweep["expected_cost"].idxmin()]
    return float(best_row["threshold"]), sweep


def full_evaluation(
    feature_table: pd.DataFrame,
    classifier: ClassifierMixin | None = None,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Fit a pipeline on the same split churn_trainer.py uses (defaults to the
    logistic baseline, same as build_pipeline()), then run the full
    evaluation suite on the held-out test set: default-threshold metrics,
    calibration, a threshold sweep, and the cost-weighted operating
    threshold. One-stop entry point for the __main__ report below (and for
    whatever explainability/dashboard code wants these numbers later).
    """
    X_train, X_test, y_train, y_test = split_feature_table(
        feature_table, test_size=test_size, random_state=random_state
    )
    pipeline = build_pipeline(random_state=random_state, classifier=classifier)
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    default_metrics = {
        **compute_metrics(y_test, y_proba),
        **confusion_counts(y_test, y_proba),
    }
    best_threshold, sweep = pick_cost_weighted_threshold(y_test, y_proba, cost_fp, cost_fn)
    cost_weighted_metrics = {
        **compute_metrics(y_test, y_proba, threshold=best_threshold),
        **confusion_counts(y_test, y_proba, threshold=best_threshold),
    }

    return {
        "pipeline": pipeline,
        "default_threshold_metrics": default_metrics,
        "cost_weighted_threshold": best_threshold,
        "cost_weighted_metrics": cost_weighted_metrics,
        "threshold_sweep": sweep,
        "calibration": calibration_table(y_test, y_proba),
    }


def save_evaluation_report(result: dict, output_dir: Path = MODEL_DIR) -> dict[str, Path]:
    """
    Write full_evaluation()'s numbers to disk next to the persisted model -
    plain JSON for the scalar metrics/threshold (human-readable, no pickle
    involved), CSV for the two tables - so milestone 10+ code (and a human
    skimming the folder) can see exactly how the saved model performed
    without retraining it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "default_threshold_metrics": result["default_threshold_metrics"],
        "cost_weighted_threshold": result["cost_weighted_threshold"],
        "cost_weighted_metrics": result["cost_weighted_metrics"],
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    sweep_path = output_dir / "threshold_sweep.csv"
    result["threshold_sweep"].to_csv(sweep_path, index=False)

    calibration_path = output_dir / "calibration.csv"
    result["calibration"].to_csv(calibration_path, index=False)

    return {"report": report_path, "threshold_sweep": sweep_path, "calibration": calibration_path}


if __name__ == "__main__":
    from churn.churn_trainer import build_random_forest_classifier, save_pipeline
    from churn.features import build_feature_table

    feature_table = build_feature_table()
    # milestone 8 settled on the random forest (better metrics + exact SHAP
    # explanations) - this is the model that actually gets persisted and
    # scored against in production, not the logistic baseline
    result = full_evaluation(feature_table, classifier=build_random_forest_classifier())

    print(f"evaluation on {len(feature_table)} policies (held-out test split, random forest)\n")

    print(f"-- default threshold ({DEFAULT_THRESHOLD}) --")
    for name, value in result["default_threshold_metrics"].items():
        print(f"  {name:<15} {value}")

    print(f"\n-- cost-weighted threshold (fp={DEFAULT_COST_FALSE_POSITIVE}, fn={DEFAULT_COST_FALSE_NEGATIVE}) --")
    print(f"  chosen threshold: {result['cost_weighted_threshold']:.2f}")
    for name, value in result["cost_weighted_metrics"].items():
        print(f"  {name:<15} {value}")

    print("\n-- calibration (predicted vs. observed churn rate by bucket) --")
    print(result["calibration"].to_string(index=False))

    model_path = save_pipeline(result["pipeline"])
    report_paths = save_evaluation_report(result)
    print(f"\nsaved model to {model_path}")
    for name, path in report_paths.items():
        print(f"saved {name} to {path}")
