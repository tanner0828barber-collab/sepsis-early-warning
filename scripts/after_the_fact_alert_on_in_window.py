"""A second count of timely alerts: was the alert ON in the 12 hours before onset? AFTER-THE-FACT.

"Timely" in this project counts only a patient's FIRST alert. A patient whose
first alert came 30 hours early and whose alert was still on at onset is
"very early," not timely. This script counts the other view: sepsis patients
with the alert on in at least one recorded hour from 12 hours before onset
through the onset hour.

Nothing is retrained and nothing is chosen. It reads the frozen models and
frozen thresholds and counts the same alerts a different way. It does not
write results/frozen.json, models/, or results/b_results.json. The plan was
written in DECISIONS.md before this ran.

Needs data/features_A.parquet and data/features_B.parquet, which the notebook
creates. Writes results/after_the_fact_alert_on_in_window.csv.

Run it:  .venv/bin/python scripts/after_the_fact_alert_on_in_window.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import REWARD_WINDOW, patient_outcomes  # noqa: E402
from sirs import sirs_alert  # noqa: E402
from split import load_piles  # noqa: E402
from train import score_rows  # noqa: E402

RESAMPLES, SEED = 1000, 2026   # the same resampling as src/bootstrap.py
MODELS = ["T0", "M1", "M2", "M3"]
NAMES = {
    "Modified SIRS": "Modified SIRS rule", "T0": "T0: hours in the unit only", "M1": "M1: body signals and trends",
    "M2": "M2: M1 plus time", "M3": "M3: the full model",
}

frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
smooth = frozen.get("smooth", 1)

features_a = pd.read_parquet(ROOT / "data" / "features_A.parquet")
dev_ids = set(load_piles().query("pile == 'dev'").pid)
sites = {
    "hospital A dev (calibration set)": (features_a[features_a.pid.isin(dev_ids)].reset_index(drop=True), "a_dev_results.json"),
    "hospital B": (pd.read_parquet(ROOT / "data" / "features_B.parquet"), "b_results.json"),
}


def count(rows, alert_on):
    """One row per sepsis patient: the first-alert outcome, and whether the alert was on in the window."""
    patients = patient_outcomes(rows, alert_on)
    onset = rows["pid"].map(patients["onset"]).to_numpy()
    hour = rows["ICULOS"].to_numpy()
    in_window = np.asarray(alert_on, dtype=bool) & (hour >= onset - REWARD_WINDOW) & (hour <= onset)
    patients["on_in_window"] = pd.Series(in_window).groupby(rows["pid"].to_numpy(), sort=False).any()
    return patients[patients.sepsis]


def share_range(flags):
    """95% range for a share of sepsis patients, by resampling those patients."""
    rng = np.random.default_rng(SEED)
    flags = np.asarray(flags, dtype=float)
    draws = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(RESAMPLES)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


table = []
for site, (rows, results_file) in sites.items():
    saved = json.loads((ROOT / "results" / results_file).read_text())
    alerts = {"Modified SIRS": sirs_alert(rows).to_numpy()}
    for name in MODELS:
        spec = frozen["models"][name]
        model = joblib.load(ROOT / "models" / f"{name}.joblib")
        alerts[name] = score_rows(model, rows, spec["columns"], smooth) >= spec["threshold"]

    for name, alert_on in alerts.items():
        sepsis = count(rows, alert_on)
        timely = int((sepsis.outcome == "caught").sum())
        very_early = int((sepsis.outcome == "very early").sum())
        missed = int((sepsis.outcome == "missed").sum())
        # the check: these must be the frozen counts, or the comparison means nothing
        expected = (saved[name]["caught"], saved[name]["very_early"], saved[name]["missed"])
        if (timely, very_early, missed) != expected:
            sys.exit(f"{site}, {name}: counted {(timely, very_early, missed)}, the frozen file says {expected}. Stopping.")

        on = int(sepsis.on_in_window.sum())
        early_still_on = int(((sepsis.outcome == "very early") & sepsis.on_in_window).sum())
        low, high = share_range(sepsis.on_in_window)
        table.append({
            "scored_on": site, "method": NAMES[name], "sepsis_patients": len(sepsis),
            "timely_first_alert_in_window": timely,
            "very_early_and_on_in_window": early_still_on,
            "very_early_and_off_in_window": very_early - early_still_on,
            "missed": missed,
            "alert_on_in_window": on,
            "alert_on_in_window_share": on / len(sepsis),
            "share_low_95": low, "share_high_95": high,
        })
        print(f"{site:34s} {NAMES[name]:30s} timely {timely:4d}   on in the window {on:4d} of {len(sepsis)} "
              f"({on / len(sepsis):.1%}, {low:.1%} to {high:.1%})   very early and off in the window {very_early - early_still_on}")

pd.DataFrame(table).to_csv(ROOT / "results" / "after_the_fact_alert_on_in_window.csv", index=False)
print("reproduced the frozen timely, very early and missed counts for every method at both sites")
print("wrote results/after_the_fact_alert_on_in_window.csv")
