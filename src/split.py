"""Cut hospital A into three piles of patients: fit, tune and dev.

The cut is by patient, never by row. If one patient's hours landed in two
piles, the model could recognize the patient instead of learning sepsis.

  fit   about 68%  the model learns from these
  tune  about 12%  used to pick settings and when to stop training
  dev   about 20%  used once to set the alert threshold and report results

Each pile gets the same share of sepsis patients.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2026
SPLIT_FILE = Path(__file__).resolve().parent.parent / "splits" / "hospital_A_piles.csv"


def make_piles(table, seed=SEED):
    """Assign every hospital A patient to fit, tune or dev."""
    ever_sepsis = table.groupby("pid", sort=True)["SepsisLabel"].max()
    patients = ever_sepsis.index.to_numpy()

    rest, dev = train_test_split(
        patients, test_size=0.20, random_state=seed, stratify=ever_sepsis.to_numpy()
    )
    fit, tune = train_test_split(
        rest, test_size=0.15, random_state=seed, stratify=ever_sepsis.loc[rest].to_numpy()
    )

    piles = pd.concat([
        pd.DataFrame({"pid": fit, "pile": "fit"}),
        pd.DataFrame({"pid": tune, "pile": "tune"}),
        pd.DataFrame({"pid": dev, "pile": "dev"}),
    ]).sort_values("pid").reset_index(drop=True)
    return piles


def save_piles(piles, path=SPLIT_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    piles.to_csv(path, index=False)


def load_piles(path=SPLIT_FILE):
    return pd.read_csv(path)
