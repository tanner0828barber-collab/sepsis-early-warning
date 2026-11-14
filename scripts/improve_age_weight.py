"""After-the-fact test: does focusing training on patients 40 and over help?

Hospital A only. Same 3 folds, yardstick and keep rule as scripts/improve.py.
Compares the frozen settings against two versions of the idea:
  (a) rows from patients aged 40 and over count double in training
  (b) train only on patients aged 40 and over (everyone is still scored)
Results are split by age group, because a model that gets better for one
group by getting worse for another is not an improvement.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from metrics import patient_outcomes, threshold_for_budget  # noqa: E402
from split import load_piles  # noqa: E402
from train import BUDGET, SEED, SETTINGS_GRID, make_model  # noqa: E402
from utility import utility_score  # noqa: E402

frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
settings, trees = frozen["settings"], frozen["trees"]
columns = model_columns("M3", extra=False)

features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool = features[features.pid.isin(set(piles.pid[piles.pile.isin(["fit", "tune"])]))].reset_index(drop=True)
per_patient = pool.groupby("pid", sort=False)["SepsisLabel"].max()
fold_of_patient = pd.Series(0, index=per_patient.index)
for k, (_, held_out) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(per_patient.index, per_patient.values)):
    fold_of_patient.iloc[held_out] = k
fold = pool.pid.map(fold_of_patient).to_numpy()
older = (pool["Age"] >= 40).to_numpy()


def held_out(weight=None, train_mask=None):
    scores = np.zeros(len(pool))
    for k in range(3):
        train = fold != k
        if train_mask is not None:
            train = train & train_mask
        model = make_model(settings, trees)
        model.fit(pool.loc[train, columns], pool.loc[train, "SepsisLabel"], sample_weight=None if weight is None else weight[train])
        scores[fold == k] = model.predict_proba(pool.loc[fold == k, columns])[:, 1]
    return scores


def measure(name, scores):
    threshold = threshold_for_budget(pool, scores, BUDGET)
    on = scores >= threshold
    patients = patient_outcomes(pool, on)
    patients["age"] = pool.groupby("pid", sort=False)["Age"].first()
    row = {"version": name, "utility": round(utility_score(pool, on), 4)}
    groups = {"all": patients.age >= 0, "under 40": patients.age < 40, "40 and over": patients.age >= 40, "80 and over": patients.age >= 80}
    for label, mask in groups.items():
        g = patients[mask]; s = g[g.sepsis]; n = g[~g.sepsis]
        row[f"{label}: alerted by onset"] = f"{int(s.outcome.isin(['caught', 'very early']).sum())} of {len(s)} ({100 * s.outcome.isin(['caught', 'very early']).mean():.1f}%)"
        row[f"{label}: false alarms"] = f"{100 * (n.alerts > 0).mean():.1f}%"
    return row


rows = [measure("frozen settings (baseline)", np.load(ROOT / "data" / "held_out_scores_pool.npy"))]
print("baseline done", flush=True)
rows.append(measure("40 and over count double", held_out(weight=np.where(older, 2.0, 1.0))))
print("double weight done", flush=True)
rows.append(measure("train only on 40 and over", held_out(train_mask=older)))
out = pd.DataFrame(rows).set_index("version").T
out.to_csv(ROOT / "results" / "after_the_fact_age_weight.csv")
pd.set_option("display.width", 250)
print(out.to_string())
