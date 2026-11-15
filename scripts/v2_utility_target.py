"""Version 2, step 7: train on a utility-shaped target. AFTER-THE-FACT, hospital A only.

The plan and the keep rule were written in DECISIONS.md before this ran.

The usual model learns a yes or no label. This one learns, for every hour, how
much the official score gains if the model alerts at that hour compared with
staying silent:

  never develops sepsis, or more than 12 hours before onset    -0.05
  from 12 to 6 hours before onset                              climbs 0 to 1
  from 6 hours before to 3 hours after onset                   climbs 1 to 2
  later than 3 hours after onset                               -0.05

The numbers come from the organizers' scoring rules. Before training, the
script checks them against the organizers' own scoring function.

Same inputs, settings, folds and 25% cap as the hospital A only baseline, which
is read back from its saved scores and not retrained. Hospital B is not loaded.
Writes results/after_the_fact_v2_utility_target.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import SEED, SETTINGS_GRID  # noqa: E402
from utility import compute_prediction_utility, utility_score  # noqa: E402

CAP = 0.25
OUTER_FOLDS, INNER_FOLDS, TREES = 3, 2, 200
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
KEEP_GAIN, UTILITY_SLACK = 0.015, 0.01
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


def gain_from_alerting(rows):
    """For every hour: official utility of alerting minus utility of staying silent."""
    hour = rows["ICULOS"].to_numpy(dtype=float)
    label_hour = rows["ICULOS"].where(rows["SepsisLabel"] == 1)
    onset = (label_hour.groupby(rows["pid"], sort=False).transform("min") + 6).to_numpy()   # empty for patients without sepsis
    d = hour - onset                                     # hours after onset (negative = before)
    gain = np.full(len(rows), -0.05)
    early = (d >= -12) & (d <= -6)
    gain[early] = (d[early] + 12) / 6
    late = (d > -6) & (d <= 3)
    gain[late] = (15 + d[late]) / 9
    return gain


def check_against_official():
    """Alert at one hour only, score it with the organizers' function, compare with our number."""
    labels = np.r_[np.zeros(20), np.ones(8)].astype(int)       # label starts at index 20, so onset is index 26
    fake = pd.DataFrame({"pid": "check", "ICULOS": np.arange(1, 29), "SepsisLabel": labels})
    ours = gain_from_alerting(fake)
    silent = compute_prediction_utility(labels, np.zeros(28), check_errors=False)
    for i in range(28):
        one = np.zeros(28); one[i] = 1
        official = compute_prediction_utility(labels, one, check_errors=False) - silent
        assert abs(official - ours[i]) < 1e-9, (i, official, ours[i])
    healthy = np.zeros(10, dtype=int)
    one = np.zeros(10); one[4] = 1
    assert abs(compute_prediction_utility(healthy, one, check_errors=False) - (-0.05)) < 1e-9
    say("target matches the organizers' scoring function at every hour of a test patient")


check_against_official()

features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool = features[features.pid.isin(set(piles.pid[piles.pile.isin(["fit", "tune"])]))].reset_index(drop=True)
target = gain_from_alerting(pool)
saved = pd.read_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
saved = saved[saved.hosp == "A"].reset_index(drop=True)
assert (saved.pid.to_numpy() == pool.pid.to_numpy()).all()
outer = saved.outer_fold.to_numpy()
patients = pool.groupby("pid", sort=False).agg(sepsis=("SepsisLabel", "max"))
patients["hosp"] = "A"
say("patients:", len(patients), "rows:", len(pool), "| share of hours where alerting gains:", f"{(target > 0).mean():.2%}")


def fold_map(group, n_splits, seed):
    strata = group["hosp"] + group["sepsis"].astype(str)
    result = pd.Series(0, index=group.index)
    for k, (_, held_out) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(group.index, strata)):
        result.iloc[held_out] = k
    return result


def fit_and_score(train, test):
    options = {k: v for k, v in SETTINGS.items() if k not in ("name", "class_weight")}
    model = HistGradientBoostingRegressor(max_iter=TREES, early_stopping=False, random_state=SEED, **options)
    model.fit(pool.loc[train, COLUMNS], target[train])
    return model.predict(pool.loc[test, COLUMNS])


def measure(name, alert_on):
    return {"model": name, "utility": utility_score(pool, alert_on), **summarize(patient_outcomes(pool, alert_on))}


stored = json.loads((ROOT / "results" / "after_the_fact_v2_both_hospitals.json").read_text())["cutoffs"]
base_cutoff = {c["outer_fold"]: c["value"] for c in stored if c["model"] == "A only" and c["budget"] == CAP and c["cutoff"] == "hospital A"}
base_alert = np.zeros(len(pool), dtype=bool)
for k in range(OUTER_FOLDS):
    base_alert[outer == k] = saved.score_A_only.to_numpy()[outer == k] >= base_cutoff[k]
baseline = measure("baseline: yes or no label", base_alert)
say("baseline read back:", f'utility {baseline["utility"]:.3f}', f'by onset {baseline["alerted_by_onset_share"]:.1%}', f'missed {baseline["missed"]}')

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
    say("utility target, outer fold", k + 1, "of", OUTER_FOLDS, "done")
candidate = measure("utility-shaped target", new_alert)

table = pd.DataFrame([baseline, candidate])
table.to_csv(ROOT / "results" / "after_the_fact_v2_utility_target.csv", index=False)
pd.DataFrame({"pid": pool.pid, "ICULOS": pool.ICULOS, "SepsisLabel": pool.SepsisLabel, "outer_fold": outer,
              "score_utility_target": scores, "alert_utility_target": new_alert, "alert_baseline": base_alert}).to_parquet(
    ROOT / "data" / "v2_held_out_scores_utility_target.parquet")
show = ["model", "utility", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share", "false_alarms", "false_alarm_share", "alerts_per_1000_patient_days"]
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(table[show].round(4).to_string(index=False))
gain = candidate["alerted_by_onset_share"] - baseline["alerted_by_onset_share"]
passes = gain >= KEEP_GAIN and candidate["utility"] >= baseline["utility"] - UTILITY_SLACK
say(f"alerted by onset change: {100 * gain:+.1f} points | utility change: {candidate['utility'] - baseline['utility']:+.3f} | keep rule:", "PASS" if passes else "NOT MET")
