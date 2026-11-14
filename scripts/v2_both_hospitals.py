"""Version 2, step 1: one model trained on both hospitals. AFTER-THE-FACT.

Hospital B was scored once with the frozen version 1 model before this was
written. Nothing here can replace that result. The plan and the keep rule
were written in DECISIONS.md before this ran.

What it answers: what do you get when a hospital has its own labeled data?
What it cannot answer: whether the model travels. It has seen both hospitals.

Method (nested cross-validation, by patient):
  outer   3 folds, balanced by hospital and by sepsis. Each patient is scored
          once, by a model that never trained on them.
  inner   inside each outer training set, 2 folds give held-out scores. The
          alert cutoff is set on those, so no patient is ever scored with a
          cutoff that saw them.

Two models on the same folds:
  A only   trained on hospital A patients, scored on hospital A patients
  both     trained on hospital A and B patients, scored on both

Hospital A dev patients are not loaded. The frozen files (results/frozen.json,
models/, results/b_results.json) are not read or written.

Writes results/after_the_fact_v2_both_hospitals.csv and .json, and keeps the
hourly held-out scores in data/ (not committed).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features, model_columns  # noqa: E402
from load import load_hospital  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

BUDGETS = [0.15, 0.25]          # 25% is the version 2 cap. 15% is shown for comparison with version 1
OUTER_FOLDS, INNER_FOLDS = 3, 2
TREES = 200
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


# ---- data: hospital A fit and tune patients, plus all of hospital B
features_a = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool_ids = set(piles.pid[piles.pile.isin(["fit", "tune"])])
features_a = features_a[features_a.pid.isin(pool_ids)]

b_file = ROOT / "data" / "features_B.parquet"
if b_file.exists():
    features_b = pd.read_parquet(b_file)
else:
    # The loader's lock was for the one frozen run. That run is done and committed,
    # so opening hospital B here is allowed, and everything from it is after-the-fact.
    features_b = build_features(load_hospital("B", frozen_run=True))
    features_b.to_parquet(b_file)

keep = ["pid", "hosp", "ICULOS", "SepsisLabel"] + COLUMNS
pool = pd.concat([features_a[keep], features_b[keep]], ignore_index=True)
hosp = pool["hosp"].to_numpy()
say("patients:", {h: int(pool.pid[hosp == h].nunique()) for h in ["A", "B"]}, "rows:", len(pool), "inputs:", len(COLUMNS))


def fold_map(patients, n_splits, seed):
    """Give each patient a fold number, balanced by hospital and by sepsis."""
    strata = patients["hosp"] + patients["sepsis"].astype(str)
    result = pd.Series(0, index=patients.index)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, held_out) in enumerate(splitter.split(patients.index, strata)):
        result.iloc[held_out] = k
    return result


patients = pool.groupby("pid", sort=False).agg(hosp=("hosp", "first"), sepsis=("SepsisLabel", "max"))
outer_of_patient = fold_map(patients, OUTER_FOLDS, SEED)
outer = pool.pid.map(outer_of_patient).to_numpy()


def fit_and_score(train, test):
    model = make_model(SETTINGS, TREES)
    model.fit(pool.loc[train, COLUMNS], pool.loc[train, "SepsisLabel"])
    return model.predict_proba(pool.loc[test, COLUMNS])[:, 1]


def nested(name, in_model):
    """Held-out scores and held-out cutoffs for one model.

    `in_model` marks the rows of the hospitals this model is trained and scored on.
    Returns the outer-fold scores and, for every budget, an alert yes/no per row
    under (a) one cutoff per hospital and (b) one shared cutoff.
    """
    scores = np.full(len(pool), np.nan)
    alerts = {(kind, budget): np.zeros(len(pool), dtype=bool) for kind in ["per hospital", "shared"] for budget in BUDGETS}
    cutoffs = []
    for k in range(OUTER_FOLDS):
        train, test = in_model & (outer != k), in_model & (outer == k)

        # inner folds: held-out scores for the outer training patients, used only to set cutoffs
        inner_of_patient = fold_map(patients.loc[pool.pid[train].unique()], INNER_FOLDS, SEED + 101 + k)
        inner = pool.pid.map(inner_of_patient).to_numpy()
        inner_scores = np.full(len(pool), np.nan)
        for j in range(INNER_FOLDS):
            inner_scores[train & (inner == j)] = fit_and_score(train & (inner != j), train & (inner == j))

        scores[test] = fit_and_score(train, test)

        for budget in BUDGETS:
            shared = threshold_for_budget(pool[train], inner_scores[train], budget)
            alerts[("shared", budget)][test] = scores[test] >= shared
            cutoffs.append({"model": name, "outer_fold": k, "budget": budget, "cutoff": "shared", "value": shared})
            for h in sorted(set(hosp[in_model])):
                rows_h = train & (hosp == h)
                own = threshold_for_budget(pool[rows_h], inner_scores[rows_h], budget)
                test_h = test & (hosp == h)
                alerts[("per hospital", budget)][test_h] = scores[test_h] >= own
                cutoffs.append({"model": name, "outer_fold": k, "budget": budget, "cutoff": f"hospital {h}", "value": own})
        say(name, "outer fold", k + 1, "of", OUTER_FOLDS, "done")
    return scores, alerts, cutoffs


def measure(rows_mask, alert_on):
    rows = pool[rows_mask]
    numbers = summarize(patient_outcomes(rows, alert_on[rows_mask]))
    wanted = [
        "patients", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share", "caught_share",
        "non_sepsis_patients", "false_alarms", "false_alarm_share", "false_alerts_per_true_alert",
        "alerts_per_1000_patient_days",
    ]
    return {"utility": utility_score(rows, alert_on[rows_mask]), **{k: numbers[k] for k in wanted}}


table, all_cutoffs, kept_scores = [], [], {}
for name, in_model in [("A only", hosp == "A"), ("both hospitals", np.ones(len(pool), dtype=bool))]:
    scores, alerts, cutoffs = nested(name, in_model)
    kept_scores[name] = scores
    all_cutoffs += cutoffs
    groups = [("hospital A", hosp == "A")]
    if name == "both hospitals":
        groups += [("hospital B", hosp == "B"), ("both together", in_model)]
    for (kind, budget), alert_on in alerts.items():
        if name == "A only" and kind == "shared":
            continue    # with one hospital the two kinds of cutoff are the same thing
        for group, mask in groups:
            row = {"model": name, "cutoff": kind, "cap": budget, "scored_on": group, **measure(mask, alert_on)}
            table.append(row)
            say(name, "|", kind, "|", f"{budget:.0%}", "|", group, "|", f'utility {row["utility"]:.3f}',
                f'by onset {row["alerted_by_onset_share"]:.1%}', f'missed {row["missed"]} of {row["sepsis_patients"]}',
                f'false alarms {row["false_alarm_share"]:.1%}')

pd.DataFrame(table).to_csv(ROOT / "results" / "after_the_fact_v2_both_hospitals.csv", index=False)
(ROOT / "results" / "after_the_fact_v2_both_hospitals.json").write_text(json.dumps({
    "label": "AFTER-THE-FACT. Hospital B had already been scored once with the frozen version 1 model.",
    "design": {
        "patients": "hospital A fit and tune, plus all of hospital B. Hospital A dev not loaded.",
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "trees": TREES, "settings": SETTINGS,
        "inputs": len(COLUMNS), "seed": SEED,
        "cutoffs": "set on inner-fold held-out scores inside each outer training set",
    },
    "results": table,
    "cutoffs": all_cutoffs,
}, indent=2))
pd.DataFrame({
    "pid": pool.pid, "hosp": pool.hosp, "ICULOS": pool.ICULOS, "SepsisLabel": pool.SepsisLabel, "outer_fold": outer,
    "score_A_only": kept_scores["A only"], "score_both": kept_scores["both hospitals"],
}).to_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
say("wrote results/after_the_fact_v2_both_hospitals.csv and .json")
