"""Step 1 of training: try the six candidate settings on hospital A.

Trains on the fit pile, judges on the tune pile, writes every try to
results/settings_search.csv. Hospital B is not touched.
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import model_columns  # noqa: E402
from split import load_piles  # noqa: E402
from train import pick_settings, search_settings  # noqa: E402

features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
piles = load_piles()
features = features.merge(piles, on="pid", how="left", validate="many_to_one")
fit_rows = features[features.pile == "fit"]
tune_rows = features[features.pile == "tune"]
print("fit rows", len(fit_rows), "tune rows", len(tune_rows), flush=True)

start = time.time()
results = search_settings(fit_rows, tune_rows, model_columns("M3"))
results.to_csv(ROOT / "results" / "settings_search.csv", index=False)
print("minutes", round((time.time() - start) / 60, 1))
print(results.round(4).to_string())
print("PICKED BY THE RULE:")
print(pick_settings(results).to_string())
