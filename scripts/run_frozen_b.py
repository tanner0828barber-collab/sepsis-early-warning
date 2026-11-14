"""THE ONE HOSPITAL B RUN.

This is the only code in the project that opens hospital B. It refuses to
run unless:
  1. results/frozen.json exists and is committed to git (so the thresholds
     were fixed before hospital B was seen), and
  2. it has not been run before (results/b_results.json does not exist).

It applies the frozen models and frozen thresholds to every hospital B
patient and writes the numbers. Nothing is chosen, tuned or adjusted here
using hospital B's sepsis labels. Two of the planned travel attempts use
hospital B's unlabeled measurements and scores, and are labeled as such.
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
from load import load_hospital  # noqa: E402
from report import evaluate_site, print_table  # noqa: E402
from rescale import site_norms  # noqa: E402

frozen_file = ROOT / "results" / "frozen.json"
done_file = ROOT / "results" / "b_results.json"

if done_file.exists():
    raise SystemExit("Hospital B has already been run once. The numbers are frozen. Not running again.")
if not frozen_file.exists():
    raise SystemExit("results/frozen.json does not exist. Freeze the thresholds first.")
tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "results/frozen.json"], cwd=ROOT, capture_output=True)
changed = subprocess.run(["git", "status", "--porcelain", "results/frozen.json", "models"], cwd=ROOT, capture_output=True, text=True)
if tracked.returncode != 0 or changed.stdout.strip():
    raise SystemExit("results/frozen.json and models/ must be committed, unchanged, before hospital B is opened.")
frozen_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

frozen = json.loads(frozen_file.read_text())
models = {name: joblib.load(ROOT / "models" / f"{name.replace(' ', '_')}.joblib") for name in frozen["models"]}

raw = load_hospital("B", frozen_run=True)
features = build_features(raw)
norms_B = site_norms(raw)          # hospital B's own average and spread: measurements only, no labels
norms_B.to_csv(ROOT / "results" / "site_norms_B.csv")

numbers, tables, m3_scores = evaluate_site(features, frozen, models, norms_B)
numbers["_run"] = {"frozen_commit": frozen_commit, "patients": int(raw.pid.nunique()), "rows": int(len(raw))}

done_file.write_text(json.dumps(numbers, indent=2))
tables["M3"].to_csv(ROOT / "results" / "b_patients_M3.csv")
for name in ["M1", "Modified SIRS"]:
    tables[name][["sepsis", "late_entry", "outcome", "first_alert", "onset", "hours_of_warning", "alerts", "hours"]].to_csv(
        ROOT / "results" / f"b_patients_{name.replace(' ', '_')}.csv"
    )
# hourly scores stay in data/, which is not committed; rerunning the notebook recreates them
pd.DataFrame({"pid": features.pid, "ICULOS": features.ICULOS, "SepsisLabel": features.SepsisLabel, "score_M3": m3_scores}).to_parquet(
    ROOT / "data" / "scores_B_M3.parquet"
)

print("frozen commit:", frozen_commit)
print_table({k: v for k, v in numbers.items() if not k.startswith("_")})
for k in ["M3", "Modified SIRS"]:
    print(k, "late entry:", numbers[k]["late_entry"], "| with history:", numbers[k]["with_history"])
print("M3 ranges:", {k: [round(x, 3) for x in v] for k, v in numbers["M3"]["ranges"].items()})
