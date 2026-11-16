"""Sunday night, step 1: nested scores for the both-hospitals model. AFTER-THE-FACT.

Hospital B was scored once with the frozen version 1 model before this was
written. Nothing here can replace that result. The plan was written in
DECISIONS.md before this ran.

It repeats Saturday's both-hospitals run (scripts/v2_both_hospitals.py) with
the same patients, folds, seed, inputs and settings, and saves two things
Saturday's run threw away:

  outer score      every patient scored once, by a model that never trained on them
  inner score k    for outer fold k, held-out scores for that fold's TRAINING
                   patients. A cutoff rule is chosen on these and then applied
                   to the outer fold, so no rule ever sees the patients it is
                   scored on.

Run it with --library hgb (scikit-learn, as on Saturday) or --library lgbm
(LightGBM, same settings, scores kept at 200 and 400 trees).

The frozen files (results/frozen.json, models/, results/b_results.json) are
not read or written. Hospital A dev patients are not loaded.

Writes data/v2_nested_<library>.parquet (not committed).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from split import load_piles  # noqa: E402
from train import SEED, SETTINGS_GRID, make_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--library", choices=["hgb", "lgbm"], required=True)
LIBRARY = parser.parse_args().library

OUTER_FOLDS, INNER_FOLDS = 3, 2
SETTINGS = next(s for s in SETTINGS_GRID if s["name"] == "slow careful")
COLUMNS = model_columns("M3", extra=False)
# one entry per saved score: its name and how many trees it uses
ARMS = {"hgb": {"hgb200": 200}, "lgbm": {"lgbm200": 200, "lgbm400": 400}}[LIBRARY]
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


# ---- data: the same pool as Saturday (hospital A fit and tune, all of hospital B)
features_a = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
pool_ids = set(piles.pid[piles.pile.isin(["fit", "tune"])])
features_a = features_a[features_a.pid.isin(pool_ids)]
features_b = pd.read_parquet(ROOT / "data" / "features_B.parquet")
keep = ["pid", "hosp", "ICULOS", "SepsisLabel"] + COLUMNS
pool = pd.concat([features_a[keep], features_b[keep]], ignore_index=True)
del features_a, features_b
say("patients:", int(pool.pid.nunique()), "rows:", len(pool), "inputs:", len(COLUMNS), "library:", LIBRARY)


def fold_map(patients, n_splits, seed):
    """Give each patient a fold number, balanced by hospital and by sepsis. Same as Saturday."""
    strata = patients["hosp"] + patients["sepsis"].astype(str)
    result = pd.Series(0, index=patients.index)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, held_out) in enumerate(splitter.split(patients.index, strata)):
        result.iloc[held_out] = k
    return result


patients = pool.groupby("pid", sort=False).agg(hosp=("hosp", "first"), sepsis=("SepsisLabel", "max"))
outer = pool.pid.map(fold_map(patients, OUTER_FOLDS, SEED)).to_numpy()

# the folds must be the ones Saturday's run used
saved = pd.read_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
assert (saved.pid.to_numpy() == pool.pid.to_numpy()).all()
assert (saved.outer_fold.to_numpy() == outer).all()
say("folds match Saturday's run")


def fit_and_score(train, test):
    """Train on `train` rows, return {arm name: scores for `test` rows}."""
    X, y = pool.loc[train, COLUMNS], pool.loc[train, "SepsisLabel"]
    if LIBRARY == "hgb":
        model = make_model(SETTINGS, 200)
        model.fit(X, y)
        return {"hgb200": model.predict_proba(pool.loc[test, COLUMNS])[:, 1]}
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=max(ARMS.values()),
        learning_rate=SETTINGS["learning_rate"],
        num_leaves=SETTINGS["max_leaf_nodes"],
        min_child_samples=SETTINGS["min_samples_leaf"],
        reg_lambda=SETTINGS["l2_regularization"],
        max_bin=255,
        n_jobs=4,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X, y)
    X_test = pool.loc[test, COLUMNS]
    return {arm: model.predict_proba(X_test, num_iteration=trees)[:, 1] for arm, trees in ARMS.items()}


out = pd.DataFrame({
    "pid": pool.pid, "hosp": pool.hosp, "ICULOS": pool.ICULOS, "SepsisLabel": pool.SepsisLabel, "outer_fold": outer,
})
for arm in ARMS:
    out[f"{arm}_outer"] = np.nan
    for k in range(OUTER_FOLDS):
        out[f"{arm}_inner{k}"] = np.nan

for k in range(OUTER_FOLDS):
    train, test = outer != k, outer == k
    inner_of_patient = fold_map(patients.loc[pool.pid[train].unique()], INNER_FOLDS, SEED + 101 + k)
    inner = pool.pid.map(inner_of_patient).to_numpy()
    for j in range(INNER_FOLDS):
        held = train & (inner == j)
        for arm, scores in fit_and_score(train & (inner != j), held).items():
            out.loc[held, f"{arm}_inner{k}"] = scores
        say("outer fold", k + 1, "inner fold", j + 1, "done")
    for arm, scores in fit_and_score(train, test).items():
        out.loc[test, f"{arm}_outer"] = scores
    say("outer fold", k + 1, "of", OUTER_FOLDS, "done")

if LIBRARY == "hgb":
    # the outer scores must be Saturday's scores, or this is not the same model
    gap = float(np.abs(out["hgb200_outer"].to_numpy() - saved.score_both.to_numpy()).max())
    say("largest gap from Saturday's saved scores:", gap)
    assert gap < 1e-9, gap

out.to_parquet(ROOT / "data" / f"v2_nested_{LIBRARY}.parquet")
say("wrote data/" + f"v2_nested_{LIBRARY}.parquet")
