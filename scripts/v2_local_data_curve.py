"""Version 2, step 5: how much local data does a new hospital need? AFTER-THE-FACT.

The plan was written in DECISIONS.md before this ran.

Hospital B is cut once by patient: 10,000 test patients that no arm trains on,
and 10,000 local patients to draw from. Each arm trains on hospital A's fit and
tune patients plus the first N local patients, then scores the test patients.
One more arm trains on the 10,000 local patients alone.

The cutoff for each arm is set on the test patients at the 25% cap, the same
way for every arm. That makes this a guide to the SHAPE of the curve (how fast
local data helps), not a score to quote. Hourly AUROC needs no cutoff and is
reported next to it.

Writes results/after_the_fact_v2_local_data_curve.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

CAP, TREES = 0.25, 200
STEPS = [0, 500, 1000, 2000, 5000, 10000]
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
keep = ["pid", "ICULOS", "SepsisLabel"] + COLUMNS
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


a = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
a = a[a.pid.isin(set(piles.pid[piles.pile.isin(["fit", "tune"])]))][keep]
b = pd.read_parquet(ROOT / "data" / "features_B.parquet")[keep]

ever_sepsis = b.groupby("pid", sort=True)["SepsisLabel"].max()
local_ids, test_ids = train_test_split(ever_sepsis.index, test_size=0.5, random_state=SEED, stratify=ever_sepsis.to_numpy())
local_order = np.random.default_rng(SEED).permutation(np.asarray(local_ids))     # one fixed order, so arms nest
test = b[b.pid.isin(set(test_ids))].reset_index(drop=True)
say("hospital A patients:", a.pid.nunique(), "| local pool:", len(local_ids), "| test:", test.pid.nunique(),
    "| test sepsis patients:", int(ever_sepsis.loc[test_ids].sum()))


def run(name, train):
    model = make_model(SETTINGS, TREES)
    model.fit(train[COLUMNS], train["SepsisLabel"])
    scores = model.predict_proba(test[COLUMNS])[:, 1]
    alert_on = scores >= threshold_for_budget(test, scores, CAP)      # same rule for every arm, see the note at the top
    numbers = summarize(patient_outcomes(test, alert_on))
    row = {"arm": name, "train_patients": int(train.pid.nunique()), "utility": utility_score(test, alert_on),
           "hourly_auroc": float(roc_auc_score(test["SepsisLabel"], scores)), **numbers}
    say(name, "|", f'utility {row["utility"]:.3f}', f'AUROC {row["hourly_auroc"]:.3f}', f'by onset {row["alerted_by_onset_share"]:.1%}',
        f'timely {row["caught"]}', f'missed {row["missed"]} of {row["sepsis_patients"]}', f'false alarms {row["false_alarm_share"]:.1%}')
    return row


rows = []
for n in STEPS:
    local = b[b.pid.isin(set(local_order[:n]))]
    rows.append(run(f"hospital A + {n:,} local patients", pd.concat([a, local], ignore_index=True)))
rows.append(run("10,000 local patients only, no hospital A", b[b.pid.isin(set(local_order))]))
pd.DataFrame(rows).to_csv(ROOT / "results" / "after_the_fact_v2_local_data_curve.csv", index=False)
say("wrote results/after_the_fact_v2_local_data_curve.csv")
