"""Version 2, step 4: compact signature inputs. AFTER-THE-FACT, hospital A only.

The plan and the keep rule were written in DECISIONS.md before this ran.

A signature summarizes the SHAPE of a signal over a window. For three signals
(partial SOFA, mean arterial pressure, BUN to creatinine ratio) over the last
7 hours, this adds:

  move    how far the signal moved across the window
  area    the area under that move: large when the change came early, small when late
  jumpy   the sum of squared hourly changes
  order   for each pair of signals, which one moved first (the Levy area)

That is 12 inputs. It is a compact, hand-computed, order 2 version, not the
challenge winner's full method. Every input uses the current hour and the six
before it, within one patient. Nothing looks ahead.

Test: the same nested cross-validation, folds and 25% cap as the hospital A only
baseline in scripts/v2_both_hospitals.py. The baseline is not retrained: its
held-out scores and cutoffs are read back and re-scored here as a check.

Hospital B is not loaded. Writes results/after_the_fact_v2_signatures.csv.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

CAP = 0.25
OUTER_FOLDS, INNER_FOLDS, TREES = 3, 2, 200
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
BASE_COLUMNS = model_columns("M3", extra=False)
SIGNALS = {"sofa": "sofa_partial", "map": "last_MAP", "bun_creat": "bun_creatinine_ratio"}
STEPS = 6          # six hourly changes span a 7 hour window
KEEP_GAIN, UTILITY_SLACK = 0.015, 0.01
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


def signature_inputs(rows):
    """The 12 compact signature inputs. `rows` holds whole patient records in time order."""
    pid = rows["pid"]
    changes = {}     # changes[name][k] = the hourly change that happened k hours ago (0 when unknown)
    for name, column in SIGNALS.items():
        change = rows[column].groupby(pid, sort=False).diff().fillna(0.0)   # first hour, or not yet measured: no change
        by_patient = change.groupby(pid, sort=False)
        changes[name] = [change.to_numpy()] + [by_patient.shift(k).fillna(0.0).to_numpy() for k in range(1, STEPS)]
    out = pd.DataFrame(index=rows.index)
    for name, c in changes.items():
        out[f"sig_move_{name}"] = sum(c)
        out[f"sig_area_{name}"] = sum((k + 1) * c[k] for k in range(STEPS))   # a change k hours ago has been "held" for k+1 hours
        out[f"sig_jumpy_{name}"] = sum(x * x for x in c)
    for a, b in combinations(SIGNALS, 2):
        ca, cb = changes[a], changes[b]
        # pairs of hours (earlier, later): positive when signal a moved before signal b
        out[f"sig_order_{a}_{b}"] = 0.5 * sum(
            ca[earlier] * cb[later] - cb[earlier] * ca[later]
            for later in range(STEPS) for earlier in range(later + 1, STEPS)
        )
    return out


# ---- data: hospital A fit and tune patients, with the folds the baseline used
features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool = features[features.pid.isin(set(piles.pid[piles.pile.isin(["fit", "tune"])]))].reset_index(drop=True)
signatures = signature_inputs(pool)
pool = pd.concat([pool, signatures], axis=1)
NEW_COLUMNS = BASE_COLUMNS + ["sofa_partial", "bun_creatinine_ratio"] + list(signatures.columns)

saved = pd.read_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
saved = saved[saved.hosp == "A"].reset_index(drop=True)
assert (saved.pid.to_numpy() == pool.pid.to_numpy()).all() and (saved.ICULOS.to_numpy() == pool.ICULOS.to_numpy()).all()
outer = saved.outer_fold.to_numpy()
patients = pool.groupby("pid", sort=False).agg(sepsis=("SepsisLabel", "max"))
patients["hosp"] = "A"
say("patients:", len(patients), "rows:", len(pool), "inputs:", len(BASE_COLUMNS), "->", len(NEW_COLUMNS))


def fold_map(group, n_splits, seed):
    """Same rule as scripts/v2_both_hospitals.py, so the inner folds match the baseline's."""
    strata = group["hosp"] + group["sepsis"].astype(str)
    result = pd.Series(0, index=group.index)
    for k, (_, held_out) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(group.index, strata)):
        result.iloc[held_out] = k
    return result


def fit_and_score(columns, train, test):
    model = make_model(SETTINGS, TREES)
    model.fit(pool.loc[train, columns], pool.loc[train, "SepsisLabel"])
    return model.predict_proba(pool.loc[test, columns])[:, 1]


def measure(name, alert_on):
    numbers = summarize(patient_outcomes(pool, alert_on))
    return {"model": name, "utility": utility_score(pool, alert_on), **numbers}


# ---- baseline: read back, not retrained
stored = json.loads((ROOT / "results" / "after_the_fact_v2_both_hospitals.json").read_text())["cutoffs"]
base_cutoff = {c["outer_fold"]: c["value"] for c in stored if c["model"] == "A only" and c["budget"] == CAP and c["cutoff"] == "hospital A"}
base_alert = np.zeros(len(pool), dtype=bool)
for k in range(OUTER_FOLDS):
    base_alert[outer == k] = saved.score_A_only.to_numpy()[outer == k] >= base_cutoff[k]
baseline = measure("baseline, 100 inputs", base_alert)
say("baseline read back:", f'utility {baseline["utility"]:.3f}', f'by onset {baseline["alerted_by_onset_share"]:.1%}', f'missed {baseline["missed"]}')

# ---- candidate: nested, same folds
scores = np.full(len(pool), np.nan)
new_alert = np.zeros(len(pool), dtype=bool)
for k in range(OUTER_FOLDS):
    train, test = outer != k, outer == k
    inner = pool.pid.map(fold_map(patients.loc[pool.pid[train].unique()], INNER_FOLDS, SEED + 101 + k)).to_numpy()
    inner_scores = np.full(len(pool), np.nan)
    for j in range(INNER_FOLDS):
        inner_scores[train & (inner == j)] = fit_and_score(NEW_COLUMNS, train & (inner != j), train & (inner == j))
    scores[test] = fit_and_score(NEW_COLUMNS, train, test)
    cutoff = threshold_for_budget(pool[train], inner_scores[train], CAP)
    new_alert[test] = scores[test] >= cutoff
    say("candidate outer fold", k + 1, "of", OUTER_FOLDS, "done")
candidate = measure(f"with signatures, {len(NEW_COLUMNS)} inputs", new_alert)

table = pd.DataFrame([baseline, candidate])
table.to_csv(ROOT / "results" / "after_the_fact_v2_signatures.csv", index=False)
show = ["model", "utility", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share", "false_alarms", "false_alarm_share"]
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(table[show].round(4).to_string(index=False))
gain = candidate["alerted_by_onset_share"] - baseline["alerted_by_onset_share"]
passes = gain >= KEEP_GAIN and candidate["utility"] >= baseline["utility"] - UTILITY_SLACK
say(f"alerted by onset change: {100 * gain:+.1f} points | utility change: {candidate['utility'] - baseline['utility']:+.3f} | keep rule:", "PASS" if passes else "NOT MET")
