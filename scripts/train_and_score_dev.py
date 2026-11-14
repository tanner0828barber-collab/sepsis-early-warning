"""Train the final models and score hospital A dev.

Reads the choices made by cross-validation in the improvement round
(results/improvement_choice.json): which inputs, which settings, how many
trees, which training target, and how much alert smoothing.

Trains T0, M1, M2 and M3 (and the rescaled variant of M3) on the fit and tune
patients together, with identical settings, sets each threshold on dev at the
15% alert budget, and writes:

  models/<name>.joblib            the trained models
  results/a_dev_candidate.json    the thresholds and settings that would be frozen
  results/a_dev_results.json      every reported number on hospital A dev
  results/a_dev_patients_M3.csv   one row per dev patient: outcome and alert hour

This is the second and last look at dev. The first look was the first model,
kept in results/first_model_a_dev_results.json. Nothing here was chosen on
dev except the threshold. Hospital B is not touched.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from load import load_hospital  # noqa: E402
from metrics import patient_outcomes, threshold_for_budget  # noqa: E402
from report import evaluate_site, print_table  # noqa: E402
from rescale import rescale_features, site_norms  # noqa: E402
from split import load_piles  # noqa: E402
from train import BUDGET, SETTINGS_GRID, fit_model, score_rows, wide_target  # noqa: E402

OTHER_BUDGETS = [0.05, 0.10, 0.20]

choice = json.loads((ROOT / "results" / "improvement_choice.json").read_text())
settings = next(s for s in SETTINGS_GRID if s["name"] == choice["settings"])
trees, smooth, extra = choice["trees"], choice["smooth"], choice["extra_inputs"]
print("choice:", choice, flush=True)

piles = load_piles()
features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
features = features.merge(piles, on="pid", how="left", validate="many_to_one")
fit_rows = features[features.pile.isin(["fit", "tune"])].reset_index(drop=True)
dev_rows = features[features.pile == "dev"].reset_index(drop=True)
print("training patients", fit_rows.pid.nunique(), "dev patients", dev_rows.pid.nunique(), flush=True)

# hospital A's own average and spread, from the training patients' raw measurements
raw = load_hospital("A")
norms_A = site_norms(raw[raw.pid.isin(set(fit_rows.pid))])
(ROOT / "results").mkdir(exist_ok=True)
norms_A.to_csv(ROOT / "results" / "site_norms_A.csv")

candidate = {
    "budget": BUDGET, "settings": settings, "trees": trees, "smooth": smooth,
    "extra_inputs": extra, "wide_target": choice["wide_target"],
    "trained_on": "hospital A fit and tune patients", "models": {}, "other_budgets": {},
}
models = {}
(ROOT / "models").mkdir(exist_ok=True)

for name in ["T0", "M1", "M2", "M3", "M3 rescaled"]:
    columns = model_columns("M3" if name == "M3 rescaled" else name, extra=extra)
    train_rows, check_rows = fit_rows, dev_rows
    if name == "M3 rescaled":
        train_rows = rescale_features(fit_rows, norms_A)
        check_rows = rescale_features(dev_rows, norms_A)
    target = wide_target(train_rows) if choice["wide_target"] else None
    model = fit_model(train_rows, columns, settings, trees, target)
    joblib.dump(model, ROOT / "models" / f"{name.replace(' ', '_')}.joblib")
    models[name] = model

    scores = score_rows(model, check_rows, columns, smooth)
    threshold = threshold_for_budget(dev_rows, scores, BUDGET)
    candidate["models"][name] = {"threshold": threshold, "columns": columns}

    if name == "M3":
        alerted = patient_outcomes(dev_rows, scores >= threshold).alerts > 0
        candidate["share_of_all_patients_alerted"] = float(alerted.mean())
        for budget in OTHER_BUDGETS:
            candidate["other_budgets"][str(budget)] = threshold_for_budget(dev_rows, scores, budget)
    print("trained:", name, flush=True)

(ROOT / "results" / "a_dev_candidate.json").write_text(json.dumps(candidate, indent=2))

numbers, tables, _ = evaluate_site(dev_rows, candidate, models, norms_A)
(ROOT / "results" / "a_dev_results.json").write_text(json.dumps(numbers, indent=2))
tables["M3"].to_csv(ROOT / "results" / "a_dev_patients_M3.csv")

print_table(numbers)
for k in ["M3", "Modified SIRS"]:
    print(k, "late entry:", numbers[k]["late_entry"], "| with history:", numbers[k]["with_history"])
print("M3 ranges:", {k: [round(x, 3) for x in v] for k, v in numbers["M3"]["ranges"].items()})
