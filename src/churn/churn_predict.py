"""
Batch scoring: load the persisted model, score every currently-active
policy, and write the results into tbl_policy_risk_score.

Only Active policies get scored (filter_active_policies()) - a Cancelled
or Expired policy already has its outcome, so "churn risk" for one isn't
a forward-looking prediction, it's the model just recognizing its own
training label after the fact. Scoring them anyway (the original design)
meant most of a batch's "at risk" flags pointed at customers who'd already
left - not actionable, and confusing for whoever reads the report.

One row per scoring run per policy (keyed on policy_num + scored_at, not
just policy_num) so re-running this over time builds a risk trend the
milestone 13 report can show, instead of only ever having the latest
number. Each row bundles everything the plan's business questions need:
churn_probability (who), top_factors from SHAP (why - milestone 8),
segment_label from the unsupervised clustering (what behaviors - milestone
5), and recommended_action from recommend.py's driver -> action lookup
(what to do about it - milestone 12).

Also writes an aggregate feature-importance CSV to models/ each run
(compute_batch_feature_importance() + save_feature_importance()) - the
"what behaviors drive risk overall" view milestone 13's report reads,
computed once here rather than recomputing SHAP on every report run.

Usage:
    python -m churn.churn_predict
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy.engine import Engine

from churn.churn_db import get_engine
from churn.churn_trainer import CATEGORICAL_FEATURES, MODEL_DIR, load_pipeline
from churn.explain import CHURN_CLASS_INDEX, aggregate_feature_importance, compute_shap_values, explain_policy
from churn.features import BEHAVIOR_FEATURE_COLUMNS, build_feature_table
from churn.recommend import recommend_action
from churn.seed_db import upsert_dataframe
from churn.segmentation import cluster_policies, label_clusters_by_risk

TOP_FACTORS_PER_POLICY = 3
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
ACTIVE_STATUS = "Active"


def filter_active_policies(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Only Active policies are worth scoring - a Cancelled/Expired one's
    outcome already happened, so a "churn risk" for it isn't a prediction,
    it's the model recognizing its own training label. Keeping this as a
    separate, pure function (rather than a query filter) so it's testable
    without a live db and so the reasoning has one clear place to live.
    """
    return feature_table[feature_table["policy_status"] == ACTIVE_STATUS].reset_index(drop=True)


def _row_top_factors(shap_values: np.ndarray, X_transformed: pd.DataFrame, row_position: int, top_n: int) -> list[dict]:
    """
    One policy's top SHAP contributors as plain JSON-serializable dicts -
    matches the shape tbl_policy_risk_score.top_factors documents:
    [{"feature": ..., "value": ...}, ...].
    """
    top = explain_policy(shap_values, X_transformed, row_position, top_n=top_n)
    return [
        {"feature": row.feature, "value": round(float(row.shap_value), 4)}
        for row in top.itertuples(index=False)
    ]


def score_all_policies(
    pipeline: Pipeline, feature_table: pd.DataFrame, top_n_factors: int = TOP_FACTORS_PER_POLICY
) -> pd.DataFrame:
    """
    Score every policy in feature_table: churn probability, SHAP top
    factors, and a segment label - one row per policy, all sharing the same
    scored_at so a single run reads back as one batch later.
    """
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    churn_probability = pipeline.predict_proba(X)[:, CHURN_CLASS_INDEX]

    shap_values, X_transformed = compute_shap_values(pipeline, X)
    top_factors = [_row_top_factors(shap_values, X_transformed, i, top_n_factors) for i in range(len(X))]
    recommended_actions = [recommend_action(factors) for factors in top_factors]

    # clustering purely for a segment label here - same feature_table already
    # in memory, no need to hit the db again via segmentation.build_segments()
    segmented = label_clusters_by_risk(cluster_policies(feature_table))

    # naive UTC timestamp - tbl_policy_risk_score.scored_at is TIMESTAMP
    # (no time zone), and every row in this batch shares the same value
    scored_at = datetime.now(timezone.utc).replace(tzinfo=None)

    return pd.DataFrame(
        {
            "policy_num": feature_table["policy_num"].to_numpy(),
            "scored_at": scored_at,
            "churn_probability": churn_probability.round(4),
            "segment_label": segmented["segment_label"].to_numpy(),
            "top_factors": top_factors,
            "recommended_action": recommended_actions,
        }
    )


def write_risk_scores(engine: Engine, scores: pd.DataFrame) -> int:
    """Persist scored rows - reuses seed_db's generic upsert-with-ON-CONFLICT-DO-NOTHING helper."""
    return upsert_dataframe(engine, "tbl_policy_risk_score", scores)


def compute_batch_feature_importance(pipeline: Pipeline, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SHAP feature importance across the whole batch - a separate
    call from score_all_policies() since the aggregate view needs the full
    per-feature SHAP matrix, not just each policy's top few factors (which
    silently drop the small contributors score_all_policies() doesn't keep).
    """
    X = feature_table[BEHAVIOR_FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    shap_values, X_transformed = compute_shap_values(pipeline, X)
    return aggregate_feature_importance(shap_values, X_transformed)


def save_feature_importance(importance: pd.DataFrame, path: Path = FEATURE_IMPORTANCE_PATH) -> Path:
    """Write the aggregate importance ranking to disk so the report can read it without touching sklearn/shap at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return path


def run_batch_scoring(engine: Engine | None = None) -> pd.DataFrame:
    """Full pipeline: load the persisted model, score every Active policy, write results to postgres + models/."""
    engine = engine or get_engine()
    pipeline = load_pipeline()
    feature_table = filter_active_policies(build_feature_table(engine))

    scores = score_all_policies(pipeline, feature_table)
    inserted = write_risk_scores(engine, scores)

    importance = compute_batch_feature_importance(pipeline, feature_table)
    importance_path = save_feature_importance(importance)

    print(f"scored {len(scores)} active policies -> {inserted} new rows in tbl_policy_risk_score "
          f"(scored_at {scores['scored_at'].iloc[0]})")
    print(f"saved aggregate feature importance to {importance_path}")
    return scores


if __name__ == "__main__":
    run_batch_scoring()
