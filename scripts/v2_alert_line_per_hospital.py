"""Version 2, step 2: the frozen model with its own alert line at each hospital. AFTER-THE-FACT.

This is a second look at hospital B, made after the one frozen run, and it is
labeled that way everywhere. The plan was written in DECISIONS.md before this ran.

The model is the frozen version 1 model. It is not retrained, and nothing in
results/frozen.json, models/ or results/b_results.json is changed.

  1. Hospital A dev: set the cutoff at the 25% cap (the version 1 rule, new cap).
     Note what share of ALL hospital A dev patients that cutoff alerts on.
  2. Hospital B: the first 2,000 patients by ID are a burn-in group. From their
     scores alone, no sepsis labels, find the cutoff that alerts on that same
     share of them. Score the other 18,000 patients with it.
  3. For comparison, score the same 18,000 with hospital A's cutoff unchanged.

Writes results/after_the_fact_v2_alert_line.csv.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from report import threshold_for_share_of_all  # noqa: E402
from split import load_piles  # noqa: E402
from train import score_rows  # noqa: E402
from utility import utility_score  # noqa: E402

CAP = 0.25
BURN_IN = 2000

frozen = json.loads((ROOT / "results" / "frozen.json").read_text())      # read only
model = joblib.load(ROOT / "models" / "M3.joblib")                        # read only
columns = frozen["models"]["M3"]["columns"]

# ---- hospital A dev: the cutoff at the 25% cap, and the share of all patients it alerts on
features_a = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
dev = features_a[features_a.pid.isin(set(piles.pid[piles.pile == "dev"]))].reset_index(drop=True)
dev_scores = score_rows(model, dev, columns, frozen.get("smooth", 1))
cutoff_a = threshold_for_budget(dev, dev_scores, CAP)
top = pd.Series(dev_scores).groupby(dev.pid.to_numpy(), sort=False).max()
share_alerted = float((top >= cutoff_a).mean())
print(f"hospital A dev: cutoff at the {CAP:.0%} cap = {cutoff_a:.5f}, alerts on {share_alerted:.1%} of all patients")

# ---- hospital B: the frozen model's hourly scores, saved by the one frozen run
b = pd.read_parquet(ROOT / "data" / "scores_B_M3.parquet")
ids = sorted(b.pid.unique())
burn_ids = set(ids[:BURN_IN])
burn, rest = b[b.pid.isin(burn_ids)], b[~b.pid.isin(burn_ids)].reset_index(drop=True)
cutoff_b = threshold_for_share_of_all(burn, burn.score_M3.to_numpy(), share_alerted)   # no labels used
print(f"hospital B: burn-in cutoff from the first {BURN_IN:,} patients = {cutoff_b:.5f}")


def measure(name, where, rows, cutoff):
    alert_on = rows.score_M3.to_numpy() >= cutoff
    numbers = summarize(patient_outcomes(rows, alert_on))
    return {"alert_line": name, "scored_on": where, "cutoff": cutoff, "utility": utility_score(rows, alert_on), **numbers}


dev_frame = dev[["pid", "ICULOS", "SepsisLabel"]].assign(score_M3=dev_scores)
table = pd.DataFrame([
    measure("hospital A cutoff at the 25% cap", "hospital A dev (calibration set)", dev_frame, cutoff_a),
    measure("hospital A cutoff, unchanged", "hospital B, the 18,000 after burn-in", rest, cutoff_a),
    measure("hospital B's own line from burn-in, no labels", "hospital B, the 18,000 after burn-in", rest, cutoff_b),
    measure("hospital A cutoff, unchanged", "hospital B, all 20,000", b.reset_index(drop=True), cutoff_a),
])
table.to_csv(ROOT / "results" / "after_the_fact_v2_alert_line.csv", index=False)
show = ["alert_line", "scored_on", "utility", "sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share",
        "false_alarms", "non_sepsis_patients", "false_alarm_share", "alerts_per_1000_patient_days"]
with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 50):
    print(table[show].round(3).to_string(index=False))
