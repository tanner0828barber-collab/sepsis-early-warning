"""Version 2, step 3: run it backwards, once. Train on hospital B, test on hospital A. AFTER-THE-FACT.

The plan was written in DECISIONS.md before this ran. It mirrors version 1:

  hospital B   cut by patient, 80% to train, 20% to set the cutoff
  hospital A   all 20,336 patients, scored once with that cutoff

Same 100 inputs, same settings, 200 trees. Nothing is tuned on hospital A.
Writes results/after_the_fact_v2_reverse.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

TREES = 200
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
keep = ["pid", "ICULOS", "SepsisLabel"] + COLUMNS

b = pd.read_parquet(ROOT / "data" / "features_B.parquet")[keep]     # built by scripts/v2_both_hospitals.py
a = pd.read_parquet(ROOT / "data" / "features_A.parquet")[keep].reset_index(drop=True)

ever_sepsis = b.groupby("pid", sort=True)["SepsisLabel"].max()
train_ids, dev_ids = train_test_split(ever_sepsis.index, test_size=0.20, random_state=SEED, stratify=ever_sepsis.to_numpy())
train = b[b.pid.isin(set(train_ids))]
dev = b[b.pid.isin(set(dev_ids))].reset_index(drop=True)
print("hospital B train patients:", train.pid.nunique(), "| cutoff patients:", dev.pid.nunique(), "| hospital A patients:", a.pid.nunique())

model = make_model(SETTINGS, TREES)
model.fit(train[COLUMNS], train["SepsisLabel"])
dev_scores = model.predict_proba(dev[COLUMNS])[:, 1]
a_scores = model.predict_proba(a[COLUMNS])[:, 1]

rows = []
for budget in [0.15, 0.25]:
    cutoff = threshold_for_budget(dev, dev_scores, budget)
    for where, frame, scores in [("hospital B, the 20% used to set the cutoff (calibration set)", dev, dev_scores), ("hospital A, all patients", a, a_scores)]:
        alert_on = scores >= cutoff
        rows.append({"budget": budget, "scored_on": where, "cutoff": cutoff, "utility": utility_score(frame, alert_on),
                     **summarize(patient_outcomes(frame, alert_on))})
table = pd.DataFrame(rows)
table.to_csv(ROOT / "results" / "after_the_fact_v2_reverse.csv", index=False)
show = ["budget", "scored_on", "utility", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share",
        "false_alarms", "non_sepsis_patients", "false_alarm_share", "alerts_per_1000_patient_days"]
with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 62):
    print(table[show].round(3).to_string(index=False))
