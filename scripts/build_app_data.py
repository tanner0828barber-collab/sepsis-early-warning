"""Pick the saved patients for the Replay tab and collect the scoreboard numbers.

Patients are chosen by a written rule, never by eye:

  Typical catch   the caught patient with the median hours of warning
                  (ties broken by lowest patient ID)
  One we miss     the first missed patient by ID, among those with history
  A false alarm   the first false alarm by ID
  Late entry      the first late-entry case by ID

Usage:
  python scripts/build_app_data.py dev   # development preview on hospital A dev
  python scripts/build_app_data.py B     # after the one frozen hospital B run
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load import load_hospital  # noqa: E402

source = sys.argv[1] if len(sys.argv) > 1 else "dev"
if source == "dev":
    patients = pd.read_csv(ROOT / "results" / "a_dev_patients_M3.csv", index_col="pid")
    raw = load_hospital("A")
    site_name = "Hospital A dev (development preview)"
elif source == "B":
    patients = pd.read_csv(ROOT / "results" / "b_patients_M3.csv", index_col="pid")
    raw = load_hospital("B", frozen_run=True)
    site_name = "Hospital B"
else:
    raise SystemExit("say dev or B")

patients = patients.sort_index()
caught = patients[(patients.outcome == "caught") & ~patients.late_entry]
median_warning = caught.hours_of_warning.median()
typical = (caught.hours_of_warning - median_warning).abs().sort_index().idxmin()
missed = patients[(patients.outcome == "missed") & ~patients.late_entry].index[0]
false_alarm = patients[patients.outcome == "false alarm"].index[0]
late_entry = patients[patients.sepsis & patients.late_entry].index[0]

cases = [
    {"label": "Typical catch", "pid": typical,
     "rule": f"the caught patient whose warning time is closest to the median ({median_warning:g} hours)"},
    {"label": "One we miss", "pid": missed, "rule": "the first missed sepsis patient by ID"},
    {"label": "A false alarm", "pid": false_alarm, "rule": "the first false alarm by ID"},
    {"label": "Late entry", "pid": late_entry,
     "rule": "the first patient by ID whose sepsis label is already on in their first recorded hour"},
]
for case in cases:
    row = patients.loc[case["pid"]]
    case["outcome"] = row["outcome"]
    case["sepsis"] = bool(row["sepsis"])

out = ROOT / "app" / "data"
out.mkdir(parents=True, exist_ok=True)
keep = raw[raw.pid.isin([c["pid"] for c in cases])].drop(columns=["hosp"])
keep.to_csv(out / "patients.csv", index=False)
(out / "cases.json").write_text(json.dumps({"site": site_name, "source": source, "cases": cases}, indent=2))
print(json.dumps(cases, indent=2))
print("rows saved:", len(keep))
