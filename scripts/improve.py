"""The improvement round: test each idea by cross-validation on hospital A.

The plan and the keep rule were written in DECISIONS.md before this ran.

  Data      fit and tune patients together, split into 3 folds by patient.
            Each patient is scored by a model that never trained on them.
  Yardstick share of sepsis patients alerted at or before onset, at the 15% budget
  Keep rule a change stays only if it adds at least 1.5 points to the yardstick
            and does not lower the utility score by more than 0.01

Dev patients are not loaded here. Hospital B is not touched.
Writes results/improvement_log.csv and results/improvement_choice.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features, model_columns  # noqa: E402
from load import load_hospital  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import BUDGET, SEED, SETTINGS_GRID, make_model, smooth_scores, wide_target  # noqa: E402
from utility import utility_score  # noqa: E402

KEEP_GAIN = 0.015
UTILITY_SLACK = 0.01
TIE_WINDOW = 0.005
TREE_STEPS = list(range(50, 601, 50))
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:4.1f} min]", *parts, flush=True)


# ---- data: fit and tune patients only
features = build_features(load_hospital("A"))
features.to_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool_ids = set(piles.pid[piles.pile.isin(["fit", "tune"])])
pool = features[features.pid.isin(pool_ids)].reset_index(drop=True)
say("pool patients", pool.pid.nunique(), "rows", len(pool))

# ---- 3 folds, by patient, same share of sepsis patients in each
per_patient = pool.groupby("pid", sort=False)["SepsisLabel"].max()
fold_of_patient = pd.Series(0, index=per_patient.index)
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for k, (_, held_out) in enumerate(splitter.split(per_patient.index, per_patient.values)):
    fold_of_patient.iloc[held_out] = k
fold = pool.pid.map(fold_of_patient).to_numpy()


def held_out_scores(columns, settings, tree_steps, target=None):
    """Score every pool patient with a model that did not train on them.

    Returns {number of trees: scores}, one entry per requested stopping point.
    """
    target = pool["SepsisLabel"] if target is None else target
    scores = {trees: np.zeros(len(pool)) for trees in tree_steps}
    for k in range(3):
        train, test = fold != k, fold == k
        model = make_model(settings, max(tree_steps))
        model.fit(pool.loc[train, columns], target[train])
        stages = model.staged_predict_proba(pool.loc[test, columns])
        for trees, probabilities in enumerate(stages, start=1):
            if trees in scores:
                scores[trees][test] = probabilities[:, 1]
    return scores


def measure(scores):
    """The yardstick and its companions for one set of held-out scores."""
    threshold = threshold_for_budget(pool, scores, BUDGET)
    alert_on = scores >= threshold
    numbers = summarize(patient_outcomes(pool, alert_on))
    return {
        "alerted_by_onset": numbers["alerted_by_onset_share"],
        "caught_in_window": numbers["caught_share"],
        "utility": utility_score(pool, alert_on),
        "hourly_auroc": float(roc_auc_score(pool["SepsisLabel"], scores)),
    }


def passes(new, old):
    return (
        new["alerted_by_onset"] - old["alerted_by_onset"] >= KEEP_GAIN
        and new["utility"] >= old["utility"] - UTILITY_SLACK
    )


log = []


def record(step, what, numbers, kept):
    log.append({"step": step, "what": what, **numbers, "kept": kept})
    pd.DataFrame(log).to_csv(ROOT / "results" / "improvement_log.csv", index=False)
    say(step, "|", what, "|", {k: round(v, 4) for k, v in numbers.items()}, "| kept:", kept)


settings_by_name = {s["name"]: s for s in SETTINGS_GRID}
choice = {"extra_inputs": False, "settings": "base weighted", "trees": 150, "wide_target": False, "smooth": 1}

# ---- step 1: the current model, measured the new way
columns = model_columns("M3", extra=False)
current_scores = held_out_scores(columns, settings_by_name[choice["settings"]], [choice["trees"]])[choice["trees"]]
current = measure(current_scores)
record("1 baseline", "current inputs, base weighted, 150 trees", current, True)

# ---- step 2: the added inputs
extra_columns = model_columns("M3", extra=True)
candidate_scores = held_out_scores(extra_columns, settings_by_name[choice["settings"]], [choice["trees"]])[choice["trees"]]
candidate = measure(candidate_scores)
keep = passes(candidate, current)
record("2 inputs", f"added inputs ({len(extra_columns)} in all)", candidate, keep)
if keep:
    choice["extra_inputs"], columns, current, current_scores = True, extra_columns, candidate, candidate_scores

# ---- step 3: settings and number of trees
tries = []
all_scores = {}
for name in ["base weighted", "slow careful", "slow careful weighted"]:
    by_trees = held_out_scores(columns, settings_by_name[name], TREE_STEPS)
    for trees, scores in by_trees.items():
        numbers = measure(scores)
        tries.append({"name": name, "trees": trees, "leaves": settings_by_name[name]["max_leaf_nodes"], **numbers})
        all_scores[(name, trees)] = scores
    say("settings searched:", name)
tries = pd.DataFrame(tries)
tries.to_csv(ROOT / "results" / "improvement_settings.csv", index=False)
close = tries[tries.alerted_by_onset >= tries.alerted_by_onset.max() - TIE_WINDOW].copy()
close["size"] = close.trees * close.leaves
picked = close.sort_values(["size", "alerted_by_onset"], ascending=[True, False]).iloc[0]
candidate = {k: float(picked[k]) for k in ["alerted_by_onset", "caught_in_window", "utility", "hourly_auroc"]}
keep = passes(candidate, current)
record("3 settings", f'{picked["name"]}, {int(picked["trees"])} trees', candidate, keep)
if keep:
    choice["settings"], choice["trees"] = picked["name"], int(picked["trees"])
    current, current_scores = candidate, all_scores[(picked["name"], int(picked["trees"]))]

# ---- step 4: a wider training target (the 12 hours before onset, not only 6)
candidate_scores = held_out_scores(
    columns, settings_by_name[choice["settings"]], [choice["trees"]], target=wide_target(pool)
)[choice["trees"]]
candidate = measure(candidate_scores)
keep = passes(candidate, current)
record("4 target", "train on the 12 hours before onset", candidate, keep)
if keep:
    choice["wide_target"], current, current_scores = True, candidate, candidate_scores

# ---- step 5: alert smoothing
best_k, best, best_scores = 1, current, current_scores
for k in [2, 3]:
    smoothed = smooth_scores(pool, current_scores, k)
    candidate = measure(smoothed)
    keep = passes(candidate, current) and candidate["alerted_by_onset"] > best["alerted_by_onset"]
    record("5 smoothing", f"alert on the average of the last {k} hourly scores", candidate, keep)
    if keep:
        best_k, best, best_scores = k, candidate, smoothed
choice["smooth"] = best_k

choice["cross_validated"] = best
(ROOT / "results" / "improvement_choice.json").write_text(json.dumps(choice, indent=2))
say("FINAL CHOICE:", json.dumps(choice))
