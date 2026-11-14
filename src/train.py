"""Train the gradient boosting model and choose its settings.

Gradient boosting builds many small decision trees one after another. Each
new tree tries to fix the mistakes of the trees before it. The version used
here accepts empty cells directly, which matters because most lab values are
empty in most hours.

Settings are chosen on the tune pile, by patient:
  for each candidate setting and each number of trees, set the cutoff that
  alerts on 15% of tune patients who never develop sepsis, then count the
  share of tune sepsis patients alerted at or before onset.

The rule for picking, written down before the search was run:
  take the best score; among every candidate within 1 percentage point of
  it, keep the smallest model (fewest trees times leaves).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import alerted_by_onset_share, threshold_for_budget

SEED = 2026
BUDGET = 0.15
MAX_TREES = 600
CHECK_EVERY = 50

# The candidates, fixed before running anything.
SETTINGS_GRID = [
    {"name": "base",          "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0,  "class_weight": None},
    {"name": "base weighted", "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0,  "class_weight": "balanced"},
    {"name": "small trees",   "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 1.0,  "class_weight": None},
    {"name": "big trees",     "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 1.0,  "class_weight": None},
    {"name": "fast",          "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 50,  "l2_regularization": 0.0,  "class_weight": None},
    {"name": "slow careful",  "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 10.0, "class_weight": None},
    # added for the improvement round, after the first search (see DECISIONS.md)
    {"name": "slow careful weighted", "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 10.0, "class_weight": "balanced"},
]
# The first search used only the first six.
FIRST_SEARCH = SETTINGS_GRID[:6]


def make_model(settings, trees):
    """Build an untrained model from one row of the settings grid."""
    options = {k: v for k, v in settings.items() if k != "name"}
    return HistGradientBoostingClassifier(
        max_iter=trees,
        early_stopping=False,   # we stop by patient on the tune pile, not by row
        random_state=SEED,
        **options,
    )


def fit_model(rows, columns, settings, trees, target=None):
    """Train one model on the given rows using only the given input columns."""
    model = make_model(settings, trees)
    model.fit(rows[columns], rows["SepsisLabel"] if target is None else target)
    return model


def score_rows(model, rows, columns, smooth=1):
    """The model score for every hourly row (a ranking score, not a probability).

    With smooth above 1, the score is the average of the patient's last few
    hourly outputs, so one noisy hour cannot fire an alert by itself.
    """
    scores = model.predict_proba(rows[columns])[:, 1]
    return smooth_scores(rows, scores, smooth)


def smooth_scores(rows, scores, hours):
    """Average each patient's scores over the last `hours` hours (past only)."""
    if hours <= 1:
        return np.asarray(scores)
    series = pd.Series(np.asarray(scores))
    averaged = series.groupby(rows["pid"].to_numpy(), sort=False).rolling(hours, min_periods=1).mean()
    return averaged.reset_index(level=0, drop=True).sort_index().to_numpy()


def wide_target(rows):
    """A wider training target: 1 from 12 hours before onset, not only 6.

    The challenge label switches on 6 hours before onset. This switches it on
    6 hours earlier still, to teach the model what the run-up looks like.
    It is used for training only. Every reported number still uses the
    challenge's own label.
    """
    hour = rows["ICULOS"]
    label_hour = hour.where(rows["SepsisLabel"] == 1)
    label_start = label_hour.groupby(rows["pid"], sort=False).transform("min")
    return (hour >= label_start - 6).astype(int)


def search_settings(fit_rows, tune_rows, columns, grid=FIRST_SEARCH, budget=BUDGET):
    """Try every candidate and every stopping point. Return one row per try."""
    results = []
    for settings in grid:
        model = fit_model(fit_rows, columns, settings, MAX_TREES)
        stages = model.staged_predict_proba(tune_rows[columns])
        for trees, probabilities in enumerate(stages, start=1):
            if trees % CHECK_EVERY:
                continue
            scores = probabilities[:, 1]
            threshold = threshold_for_budget(tune_rows, scores, budget)
            results.append({
                "name": settings["name"],
                "trees": trees,
                "leaves": settings["max_leaf_nodes"],
                "alerted_by_onset": alerted_by_onset_share(tune_rows, scores, threshold),
                "hourly_auprc": average_precision_score(tune_rows["SepsisLabel"], scores),
                "hourly_auroc": roc_auc_score(tune_rows["SepsisLabel"], scores),
            })
        print("searched:", settings["name"], flush=True)
    return pd.DataFrame(results)


def pick_settings(results, within=0.01):
    """Apply the written rule: near the best score, prefer the smallest model."""
    best = results["alerted_by_onset"].max()
    close = results[results["alerted_by_onset"] >= best - within].copy()
    close["size"] = close["trees"] * close["leaves"]
    close = close.sort_values(["size", "alerted_by_onset"], ascending=[True, False])
    return close.iloc[0]
