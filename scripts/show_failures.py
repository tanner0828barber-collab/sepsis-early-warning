"""Print five failures from the hospital B run so a person can read them.

Chosen by rule, not by eye, from results/b_patients_M3.csv:
  the first three missed sepsis patients by ID (with history before the label)
  the first two false alarms by ID
skipping the patients already used as examples in the app.

For each one it prints the hours around the event: the main vital signs,
a few labs, the model's score as a multiple of the alert line, and where the
label and the alert fall. Run only after the one frozen hospital B run.

Usage: python scripts/show_failures.py [B|dev]
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
from load import load_hospital  # noqa: E402
from train import score_rows  # noqa: E402

source = sys.argv[1] if len(sys.argv) > 1 else "B"
if source == "B":
    patients = pd.read_csv(ROOT / "results" / "b_patients_M3.csv", index_col="pid").sort_index()
    raw = load_hospital("B", frozen_run=True)
    frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
else:
    patients = pd.read_csv(ROOT / "results" / "a_dev_patients_M3.csv", index_col="pid").sort_index()
    raw = load_hospital("A")
    frozen = json.loads((ROOT / "results" / "a_dev_candidate.json").read_text())

model = joblib.load(ROOT / "models" / "M3.joblib")
spec = frozen["models"]["M3"]
used_file = ROOT / "app" / "data" / "cases.json"
used = {c["pid"] for c in json.loads(used_file.read_text())["cases"]} if used_file.exists() else set()

missed = [p for p in patients[(patients.outcome == "missed") & ~patients.late_entry].index if p not in used][:3]
false_alarms = [p for p in patients[patients.outcome == "false alarm"].index if p not in used][:2]

SHOW = ["HR", "Temp", "Resp", "MAP", "SBP", "O2Sat", "WBC", "Lactate", "Creatinine"]
chosen = [(p, "MISSED") for p in missed] + [(p, "FALSE ALARM") for p in false_alarms]
out = []
for number, (pid, kind) in enumerate(chosen, start=1):
    rows = raw[raw.pid == pid].reset_index(drop=True)
    inputs = build_features(rows)
    rows["score_x"] = (score_rows(model, inputs, spec["columns"], frozen.get("smooth", 1)) / spec["threshold"]).round(2)
    info = patients.loc[pid]
    print("=" * 100)
    print(f"Failure {number}: {kind}  patient {pid}  age {rows.Age.iloc[0]:.0f}  {'male' if rows.Gender.iloc[0] == 1 else 'female'}  "
          f"record hours {int(rows.ICULOS.iloc[0])} to {int(rows.ICULOS.iloc[-1])}")
    if kind == "MISSED":
        print(f"  label starts hour {int(info.label_start)}, onset hour {int(info.onset)}. "
              f"First alert: {'none' if pd.isna(info.first_alert) else 'hour ' + str(int(info.first_alert)) + ' (after onset)'}. "
              f"Highest score before onset: {rows[rows.ICULOS <= info.onset].score_x.max()} times the alert line.")
        center = int(info.label_start)
    else:
        print(f"  never developed sepsis. First alert hour {int(info.first_alert)}, {int(info.alerts)} separate alert(s). "
              f"Highest score: {rows.score_x.max()} times the alert line.")
        center = int(info.first_alert)
    window = rows[(rows.ICULOS >= center - 8) & (rows.ICULOS <= center + 6)]
    view = window[["ICULOS", "score_x"] + SHOW + ["SepsisLabel"]].rename(columns={"ICULOS": "hour", "score_x": "score/line", "SepsisLabel": "label"})
    print(view.to_string(index=False, na_rep="."))
    out.append({"number": number, "kind": kind, "pid": pid})
(ROOT / "results" / f"failures_{source}.json").write_text(json.dumps(out, indent=2))
