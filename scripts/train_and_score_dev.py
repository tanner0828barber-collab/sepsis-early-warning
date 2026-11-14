"""Step 2 of training: train the four planned models and score hospital A dev.

Uses the settings picked by the written rule from results/settings_search.csv.
Trains T0, M1, M2 and M3 on the fit pile with identical settings, sets each
model's threshold on dev at the 15% alert budget, and writes:

  models/<name>.joblib            the trained models
  results/a_dev_results.json      every reported number on hospital A dev
  results/a_dev_candidate.json    the threshold and settings that would be frozen

Hospital B is not touched. Nothing is frozen until this file's output has been
read and the freeze is committed.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate_alerts  # noqa: E402
from features import model_columns  # noqa: E402
from metrics import threshold_for_budget  # noqa: E402
from sirs import sirs_alert  # noqa: E402
from split import load_piles  # noqa: E402
from train import BUDGET, SETTINGS_GRID, fit_model, pick_settings, score_rows  # noqa: E402

OTHER_BUDGETS = [0.05, 0.10, 0.20]

features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
features = features.merge(load_piles(), on="pid", how="left", validate="many_to_one")
fit_rows = features[features.pile == "fit"]
dev_rows = features[features.pile == "dev"].reset_index(drop=True)

picked = pick_settings(pd.read_csv(ROOT / "results" / "settings_search.csv"))
settings = next(s for s in SETTINGS_GRID if s["name"] == picked["name"])
trees = int(picked["trees"])
print("settings:", settings, "trees:", trees, flush=True)

results = {}
candidate = {"budget": BUDGET, "settings": settings, "trees": trees, "models": {}}
(ROOT / "models").mkdir(exist_ok=True)

for name in ["T0", "M1", "M2", "M3"]:
    columns = model_columns(name)
    model = fit_model(fit_rows, columns, settings, trees)
    joblib.dump(model, ROOT / "models" / f"{name}.joblib")
    scores = score_rows(model, dev_rows, columns)
    threshold = threshold_for_budget(dev_rows, scores, BUDGET)
    numbers, patients = evaluate_alerts(dev_rows, scores >= threshold, scores)
    results[name] = numbers
    candidate["models"][name] = {"threshold": threshold, "columns": columns}

    if name == "M3":
        # share of ALL dev patients ever alerted, used later for the label-free site adjustment
        candidate["share_of_all_patients_alerted"] = float((patients.alerts > 0).mean())
        candidate["other_budgets"] = {}
        for budget in OTHER_BUDGETS:
            t = threshold_for_budget(dev_rows, scores, budget)
            n, _ = evaluate_alerts(dev_rows, scores >= t, scores)
            results[f"M3 at {int(budget * 100)}% budget"] = n
            candidate["other_budgets"][str(budget)] = t
        patients.to_csv(ROOT / "results" / "a_dev_patients_M3.csv")
    print("done:", name, flush=True)

results["Modified SIRS"], _ = evaluate_alerts(dev_rows, sirs_alert(dev_rows))
results["Never alert"], _ = evaluate_alerts(dev_rows, [False] * len(dev_rows))

(ROOT / "results" / "a_dev_results.json").write_text(json.dumps(results, indent=2))
(ROOT / "results" / "a_dev_candidate.json").write_text(json.dumps(candidate, indent=2))

show = [
    "sepsis_patients", "caught", "very_early", "missed", "false_alarms",
    "false_alarm_share", "alerted_by_onset_share", "median_hours_of_warning",
    "false_alerts_per_true_alert", "alerts_per_1000_patient_days", "utility",
    "hourly_auroc", "hourly_auprc",
]
table = pd.DataFrame({k: {c: v.get(c) for c in show} for k, v in results.items()}).T
pd.set_option("display.width", 250)
print(table.round(3).to_string())
for k in ["M3", "Modified SIRS"]:
    print(k, "late entry:", results[k]["late_entry"], "with history:", results[k]["with_history"])
