"""One more idea for the improvement round, added at 2:52 PM after step 2 came back.

The first model alerts by onset on 84% of sepsis patients whose label starts
at hour 24 or later, and only 34% of the earlier ones. Its score climbs with
time in the unit, so one fixed alert line is rarely crossed early in a stay.

The idea: grade each patient against other patients at the same hour of
their stay. The model's score is replaced by "what share of all patients at
this same hour score lower than this." That needs no sepsis labels.

Same data, same yardstick, same keep rule as the rest of the round.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from hour_adjust import fit_hour_table, hour_adjusted  # noqa: E402
from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import BUDGET, SEED, SETTINGS_GRID, make_model, smooth_scores, wide_target  # noqa: E402
from utility import utility_score  # noqa: E402

started = time.time()
choice = json.loads((ROOT / "results" / "improvement_choice.json").read_text())
settings = next(s for s in SETTINGS_GRID if s["name"] == choice["settings"])

features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool = features[features.pid.isin(set(piles.pid[piles.pile.isin(["fit", "tune"])]))].reset_index(drop=True)
per_patient = pool.groupby("pid", sort=False)["SepsisLabel"].max()
fold_of_patient = pd.Series(0, index=per_patient.index)
for k, (_, held_out) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(per_patient.index, per_patient.values)):
    fold_of_patient.iloc[held_out] = k
fold = pool.pid.map(fold_of_patient).to_numpy()

columns = model_columns("M3", extra=choice["extra_inputs"])
target = wide_target(pool) if choice["wide_target"] else pool["SepsisLabel"]
scores = np.zeros(len(pool))
for k in range(3):
    model = make_model(settings, choice["trees"])
    model.fit(pool.loc[fold != k, columns], target[fold != k])
    scores[fold == k] = model.predict_proba(pool.loc[fold == k, columns])[:, 1]
scores = smooth_scores(pool, scores, choice["smooth"])
np.save(ROOT / "data" / "held_out_scores_pool.npy", scores)


def measure(s):
    threshold = threshold_for_budget(pool, s, BUDGET)
    on = s >= threshold
    patients = patient_outcomes(pool, on)
    numbers = summarize(patients)
    sepsis = patients[patients.sepsis]
    early = sepsis[sepsis.label_start < 24]
    late = sepsis[sepsis.label_start >= 24]
    share = lambda g: float(g.outcome.isin(["caught", "very early"]).mean())
    return {
        "alerted_by_onset": numbers["alerted_by_onset_share"],
        "caught_in_window": numbers["caught_share"],
        "utility": utility_score(pool, on),
        "hourly_auroc": float(roc_auc_score(pool["SepsisLabel"], s)),
        "label_before_hour_24": share(early),
        "label_hour_24_or_later": share(late),
        "median_warning": numbers["median_hours_of_warning"],
    }


rows = [{"what": "chosen model, one fixed alert line", **measure(scores)}]
table = fit_hour_table(pool["ICULOS"].to_numpy(), scores)
adjusted = hour_adjusted(pool["ICULOS"].to_numpy(), scores, table)
rows.append({"what": "graded against patients at the same hour", **measure(adjusted)})
for weight in [0.5]:
    overall = pd.Series(scores).rank(pct=True).to_numpy()
    rows.append({"what": "half and half", **measure(weight * adjusted + (1 - weight) * overall)})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "improvement_hour_adjust.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string())
print("minutes", round((time.time() - started) / 60, 1))
