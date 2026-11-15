"""Version 2, step 6: a hospital B only model, nested. AFTER-THE-FACT.

The plan and the keep rule were written in DECISIONS.md before this ran.

Question: is hospital B better served by a model trained only on its own
patients than by the model trained on both hospitals?

Same outer folds as scripts/v2_both_hospitals.py, 2 inner folds to set the
cutoff at the 25% cap, same inputs and settings. The both-hospitals baseline
is not retrained: its held-out scores and cutoffs are read back and re-scored.

Writes results/after_the_fact_v2_local_only.csv.
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

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

CAP = 0.25
OUTER_FOLDS, INNER_FOLDS, TREES = 3, 2, 200
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
KEEP_GAIN, UTILITY_SLACK = 0.015, 0.01
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


pool = pd.read_parquet(ROOT / "data" / "features_B.parquet")[["pid", "ICULOS", "SepsisLabel"] + COLUMNS].reset_index(drop=True)
saved = pd.read_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
saved = saved[saved.hosp == "B"].reset_index(drop=True)
assert (saved.pid.to_numpy() == pool.pid.to_numpy()).all() and (saved.ICULOS.to_numpy() == pool.ICULOS.to_numpy()).all()
outer = saved.outer_fold.to_numpy()
patients = pool.groupby("pid", sort=False).agg(sepsis=("SepsisLabel", "max"))
say("hospital B patients:", len(patients), "rows:", len(pool))


def fold_map(group, n_splits, seed):
    result = pd.Series(0, index=group.index)
    for k, (_, held_out) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(group.index, group["sepsis"])):
        result.iloc[held_out] = k
    return result


def fit_and_score(train, test):
    model = make_model(SETTINGS, TREES)
    model.fit(pool.loc[train, COLUMNS], pool.loc[train, "SepsisLabel"])
    return model.predict_proba(pool.loc[test, COLUMNS])[:, 1]


def measure(name, alert_on):
    return {"model": name, "utility": utility_score(pool, alert_on), **summarize(patient_outcomes(pool, alert_on))}


# ---- baseline: the both-hospitals model at hospital B, read back
stored = json.loads((ROOT / "results" / "after_the_fact_v2_both_hospitals.json").read_text())["cutoffs"]
base_cutoff = {c["outer_fold"]: c["value"] for c in stored if c["model"] == "both hospitals" and c["budget"] == CAP and c["cutoff"] == "hospital B"}
base_alert = np.zeros(len(pool), dtype=bool)
for k in range(OUTER_FOLDS):
    base_alert[outer == k] = saved.score_both.to_numpy()[outer == k] >= base_cutoff[k]
baseline = measure("both-hospitals model, scored at hospital B", base_alert)
say("baseline read back:", f'utility {baseline["utility"]:.3f}', f'by onset {baseline["alerted_by_onset_share"]:.1%}', f'missed {baseline["missed"]}')

# ---- candidate: hospital B only, nested, same outer folds
scores = np.full(len(pool), np.nan)
new_alert = np.zeros(len(pool), dtype=bool)
for k in range(OUTER_FOLDS):
    train, test = outer != k, outer == k
    inner = pool.pid.map(fold_map(patients.loc[pool.pid[train].unique()], INNER_FOLDS, SEED + 101 + k)).to_numpy()
    inner_scores = np.full(len(pool), np.nan)
    for j in range(INNER_FOLDS):
        inner_scores[train & (inner == j)] = fit_and_score(train & (inner != j), train & (inner == j))
    scores[test] = fit_and_score(train, test)
    new_alert[test] = scores[test] >= threshold_for_budget(pool[train], inner_scores[train], CAP)
    say("hospital B only, outer fold", k + 1, "of", OUTER_FOLDS, "done")
candidate = measure("hospital B only model", new_alert)

table = pd.DataFrame([baseline, candidate])
table.to_csv(ROOT / "results" / "after_the_fact_v2_local_only.csv", index=False)
pd.DataFrame({"pid": pool.pid, "ICULOS": pool.ICULOS, "SepsisLabel": pool.SepsisLabel, "outer_fold": outer, "score_B_only": scores}).to_parquet(
    ROOT / "data" / "v2_held_out_scores_B_only.parquet")
show = ["model", "utility", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share", "false_alarms", "false_alarm_share", "alerts_per_1000_patient_days"]
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(table[show].round(4).to_string(index=False))
gain = candidate["alerted_by_onset_share"] - baseline["alerted_by_onset_share"]
passes = gain >= KEEP_GAIN and candidate["utility"] >= baseline["utility"] - UTILITY_SLACK
say(f"alerted by onset change: {100 * gain:+.1f} points | utility change: {candidate['utility'] - baseline['utility']:+.3f} | keep rule:", "PASS" if passes else "NOT MET")
